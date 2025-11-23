import os
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from agent_play.episode_runner import run_episode
from agent_play.memory_utils import get_process, memory_limit_hit
from agents.base_agent import BaseAgent
from game.environment import GameEnv
from game import shared_cache

def _play_episode_worker(
    env_class: type[GameEnv],
    agent_x: BaseAgent,
    agent_o: BaseAgent,
    coin_flip_start: bool,
    collect_history: bool,
    cache_context: Optional[Dict[str, Any]] = None,
    disable_shared_cache: bool = False,
    episodes_per_batch: int = 1,
    epsilon_start_x: Optional[float] = None,
    epsilon_start_o: Optional[float] = None,
) -> Tuple[Optional[List[List[Dict[str, Any]]]], Dict, Dict, Dict, Dict, int]:
    if disable_shared_cache:
        shared_cache.install_context(None)
    else:
        shared_cache.ensure_context(cache_context)

    histories: Optional[List[List[Dict[str, Any]]]] = [] if collect_history else None
    env = env_class()

    # Enable delta tracking so we can return only the updates produced in this batch.
    agent_x.enable_delta_tracking()
    agent_o.enable_delta_tracking()

    if epsilon_start_x is not None:
        agent_x.epsilon = epsilon_start_x
    if epsilon_start_o is not None:
        agent_o.epsilon = epsilon_start_o

    episodes_run = 0
    while episodes_run < episodes_per_batch:
        state_history, winner, _ = run_episode(env, agent_x, agent_o, coin_flip_start=coin_flip_start)
        agent_x.learn_result(winner, state_history)
        agent_o.learn_result(winner, state_history)

        episodes_run += 1

        if histories is not None:
            histories.append(env.history.copy())

        agent_x.epsilon = max(agent_x.min_epsilon, agent_x.epsilon * agent_x.epsilon_decay)
        agent_o.epsilon = max(agent_o.min_epsilon, agent_o.epsilon * agent_o.epsilon_decay)

    q_delta_x, visit_delta_x = agent_x.drain_deltas()
    q_delta_o, visit_delta_o = agent_o.drain_deltas()

    return histories, q_delta_x, visit_delta_x, q_delta_o, visit_delta_o, episodes_run


def _train_agents_parallel(
    env_class: type[GameEnv],
    agent_x: BaseAgent,
    agent_o: BaseAgent,
    episodes: int,
    *,
    coin_flip_start: bool,
    max_workers: Optional[int],
    show_progress: bool,
    collect_history: bool,
    memory_stop_threshold_mb: Optional[int],
    process,
    cache_context: Optional[Dict[str, Any]],
    disable_shared_cache: bool,
) -> List[List[Dict[str, Any]]]:
    def advance_epsilon(value: float, decay: float, minimum: float, steps: int) -> float:
        eps = value
        for _ in range(steps):
            eps = max(minimum, eps * decay)
        return eps

    if disable_shared_cache:
        shared_cache.install_context(None)
    else:
        shared_cache.ensure_context(cache_context)
    if episodes <= 0:
        return []

    if max_workers:
        worker_count = max(1, min(max_workers, episodes))
    else:
        cpu_guess = max(1, (os.cpu_count() or 2) - 1)
        worker_count = min(cpu_guess, episodes)

    approx = episodes // (worker_count * 4) if worker_count else episodes
    batch_size = min(64, max(1, approx))

    histories: List[List[Dict[str, Any]]] = []
    submitted = 0
    completed = 0
    stop = False

    scheduled_eps_x = agent_x.epsilon
    scheduled_eps_o = agent_o.epsilon
    completed_eps_x = agent_x.epsilon
    completed_eps_o = agent_o.epsilon

    pending: set[Future] = set()

    progress = tqdm(total=episodes, ncols=80, desc="Training (parallel)", disable=not show_progress)
    try:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            while completed < episodes and not stop:
                while submitted < episodes and len(pending) < worker_count and not stop:
                    remaining = episodes - submitted
                    current_batch = min(batch_size, remaining)
                    start_eps_x = scheduled_eps_x
                    start_eps_o = scheduled_eps_o
                    scheduled_eps_x = advance_epsilon(scheduled_eps_x, agent_x.epsilon_decay, agent_x.min_epsilon, current_batch)
                    scheduled_eps_o = advance_epsilon(scheduled_eps_o, agent_o.epsilon_decay, agent_o.min_epsilon, current_batch)

                    future = executor.submit(
                        _play_episode_worker,
                        env_class,
                        agent_x,
                        agent_o,
                        coin_flip_start,
                        collect_history,
                        cache_context,
                        disable_shared_cache,
                        current_batch,
                        start_eps_x,
                        start_eps_o,
                    )
                    pending.add(future)
                    submitted += current_batch

                if not pending:
                    break

                finished = next(as_completed(pending))
                pending.discard(finished)
                try:
                    batch_histories, q_delta_x, visit_delta_x, q_delta_o, visit_delta_o, episodes_run = finished.result()
                except BaseException:
                    stop = True
                    for leftover in list(pending):
                        leftover.cancel()
                    raise

                if collect_history and batch_histories:
                    histories.extend(batch_histories)

                agent_x.apply_deltas(q_delta_x, visit_delta_x)
                agent_o.apply_deltas(q_delta_o, visit_delta_o)

                completed += episodes_run
                progress.update(episodes_run)

                completed_eps_x = advance_epsilon(completed_eps_x, agent_x.epsilon_decay, agent_x.min_epsilon, episodes_run)
                completed_eps_o = advance_epsilon(completed_eps_o, agent_o.epsilon_decay, agent_o.min_epsilon, episodes_run)
                agent_x.epsilon = completed_eps_x
                agent_o.epsilon = completed_eps_o

                if memory_limit_hit(process, memory_stop_threshold_mb):
                    stop = True

            if stop:
                for leftover in list(pending):
                    leftover.cancel()
    finally:
        progress.close()

    agent_x.epsilon = completed_eps_x
    agent_o.epsilon = completed_eps_o

    return histories


def _train_agents_sequential(
    env,
    agent_x,
    agent_o,
    episodes: int,
    memory_stop_threshold_mb: Optional[int] = None,
    process=None,
    coin_flip_start: bool = False,
    history_path: Optional[str] = None,
    show_progress: bool = True,
) -> List[List[Dict[str, Any]]]:
    all_histories: List[List[Dict[str, Any]]] = []
    episode = 0
    with tqdm(total=episodes, ncols=80, desc="Training", disable=not show_progress) as progress_bar:
        while episode < episodes:
            if memory_limit_hit(process, memory_stop_threshold_mb):
                print("Memory threshold reached; stopping early.")
                break
            try:
                state_history, winner, stats = run_episode(env, agent_x, agent_o, coin_flip_start=coin_flip_start)
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
    disable_shared_cache: bool = False,
):
    process = get_process() if memory_stop_threshold_mb else None
    all_histories: List[List[Dict[str, Any]]] = []
    manager = None
    cache_context: Optional[Dict[str, Any]] = None

    try:
        if parallel:
            if not disable_shared_cache:
                manager = Manager()
                cache_context = shared_cache.create_shared_context(manager)
                shared_cache.install_context(cache_context)
            else:
                shared_cache.install_context(None)
            all_histories = _train_agents_parallel(
                env_class=type(env),
                agent_x=agent_x,
                agent_o=agent_o,
                episodes=episodes,
                coin_flip_start=coin_flip_start,
                max_workers=max_workers,
                show_progress=show_progress,
                collect_history=bool(history_path),
                memory_stop_threshold_mb=memory_stop_threshold_mb,
                process=process,
                cache_context=cache_context,
                disable_shared_cache=disable_shared_cache,
            )
        else:
            shared_cache.install_context(None)
            all_histories = _train_agents_sequential(
                env,
                agent_x,
                agent_o,
                episodes,
                memory_stop_threshold_mb,
                process,
                coin_flip_start,
                history_path,
                show_progress,
            )
    finally:
        shared_cache.install_context(None)
        if manager is not None:
            manager.shutdown()

    if history_path and all_histories:
        _download_training_data(all_histories, Path(history_path))
    if agent_x_save_path:
        _save_agent_snapshot(agent_x, agent_x_save_path)
    if agent_o_save_path:
        _save_agent_snapshot(agent_o, agent_o_save_path)

    return all_histories


def _download_training_data(all_histories: list[list[dict[str, Any]]], history_path: Path):
    import json
    import zipfile

    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(history_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for i, history in enumerate(all_histories):
            serialized = []
            for entry in history:
                move = entry.get("move")
                serialized.append({
                    "player": entry.get("player"),
                    "move": list(move) if move is not None else None,
                })
            json_bytes = json.dumps(serialized, ensure_ascii=False, indent=2).encode("utf-8")
            zipf.writestr(f"episode_{i + 1}.json", json_bytes)


def _save_agent_snapshot(agent: BaseAgent, path_hint: Optional[str]) -> None:
    if agent is None or not path_hint:
        return
    target_path = Path(path_hint)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(target_path))
