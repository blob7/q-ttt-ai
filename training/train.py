from typing import Optional
from pathlib import Path

from training.memory_utils import get_process, memory_limit_hit
from training.episode_runner import run_episode
from game.environment import GameEnv
from agents.base_agent import BaseAgent
from tqdm import tqdm


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
):
    """High-level orchestrator that manages training loops, memory safety, and progress output."""

    all_histories = []

    process = get_process() if memory_stop_threshold_mb else None

    progress_segments = 100 if show_progress else 0
    last_segment = -1
    progress_active = False

    episode = 0
    progress_bar = tqdm(total=episodes, ncols=80, unit="ep") if show_progress else None
    while episode < episodes:
        if memory_limit_hit(process, memory_stop_threshold_mb):
            print("Memory threshold reached before starting new episode; stopping early.")
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
            
            if progress_bar:
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

    if progress_bar:
        progress_bar.close()


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