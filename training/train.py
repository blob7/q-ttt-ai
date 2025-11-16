import os
import signal
from typing import List, Optional, Tuple
from pathlib import Path

from game.board import PlayerPiece
from training.memory_utils import get_process, memory_limit_hit
from training.episode_runner import run_episode
from game.environment import GameEnv
from agents.base_agent import BaseAgent


from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import math




# --------------------------------------------------
# Parallel training helper
# --------------------------------------------------

def _train_agents_parallel(env_class, agent_class_x, agent_class_o, episodes, coin_flip_start, max_workers=None):
    
    if max_workers is None:
        max_workers = min(4, episodes)

    chunk_size = math.ceil(episodes / max_workers)
    futures = []
    all_histories = []
    q_tables_x, q_tables_o = [], []
    visit_tables_x, visit_tables_o = [], []



    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for i in range(max_workers):
            futures.append(
                executor.submit(_run_chunk, chunk_size, env_class, agent_class_x, agent_class_o, coin_flip_start, i)
            )

        try:
            for future in as_completed(futures):
                q_x_chunk, q_o_chunk, visit_x_chunk, visit_o_chunk, chunk_histories = future.result()
                all_histories.extend(chunk_histories)
                q_tables_x.append(q_x_chunk)
                q_tables_o.append(q_o_chunk)
                visit_tables_x.append(visit_x_chunk)
                visit_tables_o.append(visit_o_chunk)

        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Terminating workers...")
            executor.shutdown(wait=False, cancel_futures=True)

    return all_histories, q_tables_x, visit_tables_x, q_tables_o, visit_tables_o


def _run_chunk(chunk_episodes, env_class, agent_class_x, agent_class_o, coin_flip_start, worker_index):
    env = env_class()
    agent_x = agent_class_x(role=PlayerPiece.X.value)
    agent_o = agent_class_o(role=PlayerPiece.O.value)
    chunk_histories = []

    with tqdm(total=chunk_episodes, ncols=80, position=worker_index, desc=f"Worker {os.getpid()}") as bar:
        for _ in range(chunk_episodes):
            state_history, winner = run_episode(env, agent_x, agent_o, coin_flip_start)
            agent_x.learn_result(winner, state_history)
            agent_o.learn_result(winner, state_history)
            chunk_histories.append(state_history)
            bar.update(1)
    # return Q-tables for merging
    return agent_x.q_values, agent_o.q_values, agent_x.visit_counts, agent_o.visit_counts, chunk_histories

# --------------------------------------------------
# Sequential training helper
# --------------------------------------------------
def _train_agents_sequential(
    env,
    agent_x,
    agent_o,
    episodes: int,
    memory_stop_threshold_mb: Optional[int] = None,
    process=None,
    coin_flip_start: bool = False,
    history_path: Optional[str] = None,
) -> List:
    """Run training sequentially in the current process."""
    all_histories = []
    episode = 0
    with tqdm(total=episodes, ncols=80, desc="Training") as progress_bar:
        while episode < episodes:
            if memory_limit_hit(process, memory_stop_threshold_mb):
                print("Memory threshold reached; stopping early.")
                break
            try:
                state_history, winner = run_episode(env, agent_x, agent_o, coin_flip_start=coin_flip_start)
                agent_x.learn_result(winner, state_history)
                agent_o.learn_result(winner, state_history)

                if history_path:
                    all_histories.append(env.history.copy())

                episode += 1
                agent_x.decay_epsilon()
                agent_o.decay_epsilon()
                progress_bar.update(1)
            except KeyboardInterrupt:
                choice = input("Training interrupted. Cancel training? [y/N]: ").strip().lower()
                if choice in ("y", "yes"):
                    print("Training cancelled by user.")
                    break
                print("Resuming training...")
            except Exception as e:
                print(f"Error on episode {episode + 1}: {e}")
                if env.history:
                    all_histories.append(env.history.copy())
                if history_path:
                    print("Saving collected match histories before exiting...")
                    Path(history_path).write_text(str(all_histories))
                raise
    return all_histories

# --------------------------------------------------
# Main training orchestrator
# --------------------------------------------------
def train_agents(
    env: GameEnv,
    agent_x: BaseAgent,
    agent_o: BaseAgent,
    episodes: int = 1000,
    *,
    memory_stop_threshold_mb: Optional[int] = None,
    history_path: Optional[str] = None,
    agent_x_save_path: Optional[str] = None,
    agent_o_save_path: Optional[str] = None,
    show_progress: bool = True,
    coin_flip_start: bool = False,
    parallel: bool = True,
    max_workers: Optional[int] = None,
):
    """High-level orchestrator for sequential or parallel training."""

    process = get_process() if memory_stop_threshold_mb else None


    if parallel:
        all_histories, q_tables_x, visit_tables_x, q_tables_o, visit_tables_o = _train_agents_parallel(
            type(env), type(agent_x), type(agent_o),
            episodes, coin_flip_start, max_workers
        )
        agent_x.merge_q_tables(q_tables_x, visit_tables_x)
        agent_o.merge_q_tables(q_tables_o, visit_tables_o)
    else:
        all_histories = _train_agents_sequential(
            env, agent_x, agent_o, episodes,
            memory_stop_threshold_mb, process,
            coin_flip_start, history_path
        )





    if history_path and all_histories:
        _download_training_data(all_histories, Path(history_path))
    if agent_x_save_path:
        _save_agent_snapshot(agent_x, agent_x_save_path)
    if agent_o_save_path:
        _save_agent_snapshot(agent_o, agent_o_save_path)

    return all_histories




def _download_training_data(all_histories: list[dict], history_path: Path):
    import zipfile
    import json

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    with zipfile.ZipFile(history_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for i, history in enumerate(all_histories):
            # Convert history to JSON string
            json_bytes = json.dumps([
                {"player": entry.get("player"), "move": list(entry.get("move"))}
                for entry in history
            ], ensure_ascii=False, indent=2).encode("utf-8")

            # Save as episode_i.json inside the ZIP
            zipf.writestr(f"episode_{i+1}.json", json_bytes)


def _save_agent_snapshot(agent: BaseAgent, path_hint: Optional[str]) -> None:
    if agent is None or not path_hint:
        return
    target_path = Path(path_hint)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(target_path))