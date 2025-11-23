import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from agent_play.episode_runner import run_episode
from agent_play.memory_utils import get_process, memory_limit_hit
from agents.base_agent import BaseAgent
from agents.shared_backend import SharedActionValueBackend
from game.environment import GameEnv


_WORKER_ENV_CLASS = None
_WORKER_AGENT_TEMPLATE_X: Optional[BaseAgent] = None
_WORKER_AGENT_TEMPLATE_O: Optional[BaseAgent] = None
_WORKER_ENV_INSTANCE: Optional[GameEnv] = None
_WORKER_AGENT_X_INSTANCE: Optional[BaseAgent] = None
_WORKER_AGENT_O_INSTANCE: Optional[BaseAgent] = None
_WORKER_COIN_FLIP = False
_WORKER_COLLECT_HISTORY = False


def _init_process_worker(env_class, agent_template_x, agent_template_o, coin_flip_start, collect_history):
    global _WORKER_ENV_CLASS, _WORKER_AGENT_TEMPLATE_X, _WORKER_AGENT_TEMPLATE_O
    global _WORKER_ENV_INSTANCE, _WORKER_AGENT_X_INSTANCE, _WORKER_AGENT_O_INSTANCE
    global _WORKER_COIN_FLIP, _WORKER_COLLECT_HISTORY

    _WORKER_ENV_CLASS = env_class
    _WORKER_AGENT_TEMPLATE_X = agent_template_x
    _WORKER_AGENT_TEMPLATE_O = agent_template_o
    _WORKER_COIN_FLIP = coin_flip_start
    _WORKER_COLLECT_HISTORY = collect_history

    _WORKER_ENV_INSTANCE = None
    _WORKER_AGENT_X_INSTANCE = None
    _WORKER_AGENT_O_INSTANCE = None


def _process_worker_chunk(schedule_x: List[float], schedule_o: List[float]) -> tuple[List[List[Dict[str, Any]]], int]:
    global _WORKER_ENV_INSTANCE, _WORKER_AGENT_X_INSTANCE, _WORKER_AGENT_O_INSTANCE

    if _WORKER_ENV_CLASS is None or _WORKER_AGENT_TEMPLATE_X is None or _WORKER_AGENT_TEMPLATE_O is None:
        raise RuntimeError("Worker not initialised")

    if _WORKER_ENV_INSTANCE is None:
        _WORKER_ENV_INSTANCE = _WORKER_ENV_CLASS()
    if _WORKER_AGENT_X_INSTANCE is None:
        _WORKER_AGENT_X_INSTANCE = _WORKER_AGENT_TEMPLATE_X.fork_shared()  # type: ignore[union-attr]
    if _WORKER_AGENT_O_INSTANCE is None:
        _WORKER_AGENT_O_INSTANCE = _WORKER_AGENT_TEMPLATE_O.fork_shared()  # type: ignore[union-attr]

    env = _WORKER_ENV_INSTANCE
    agent_x = _WORKER_AGENT_X_INSTANCE
    agent_o = _WORKER_AGENT_O_INSTANCE

    if env is None or agent_x is None or agent_o is None:
        raise RuntimeError("Worker context not prepared")

    local_histories: List[List[Dict[str, Any]]] = []

    for eps_x, eps_o in zip(schedule_x, schedule_o):
        agent_x.epsilon = eps_x
        agent_o.epsilon = eps_o

        state_history, winner, _ = run_episode(
            env,
            agent_x,
            agent_o,
            coin_flip_start=_WORKER_COIN_FLIP,
        )

        agent_x.learn_result(winner, state_history)
        agent_o.learn_result(winner, state_history)

        if _WORKER_COLLECT_HISTORY:
            local_histories.append(env.history.copy())

    return local_histories, len(schedule_x)


def _estimate_backend_capacity(episodes: int) -> int:
    estimated_states = episodes * 18
    baseline = 200_000
    cap = max(baseline, estimated_states)
    return min(cap, 1_000_000)


def _ensure_shared_backend(
    agent_x: BaseAgent,
    agent_o: BaseAgent,
    *,
    capacity: int,
    max_actions: int,
) -> None:
    shared_backend = None
    if agent_x.uses_shared_backend():
        shared_backend = agent_x.get_shared_backend()
    elif agent_o.uses_shared_backend():
        shared_backend = agent_o.get_shared_backend()

    if shared_backend is None:
        shared_backend = SharedActionValueBackend.create(capacity=capacity, max_actions=max_actions)
        agent_x.use_shared_backend(shared_backend, own=True)
        agent_o.use_shared_backend(shared_backend.fork(), own=False)
        return

    if not agent_x.uses_shared_backend():
        agent_x.use_shared_backend(shared_backend.fork(), own=False)
    if not agent_o.uses_shared_backend():
        agent_o.use_shared_backend(shared_backend.fork(), own=False)



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
    batch_size: Optional[int],
) -> List[List[Dict[str, Any]]]:
    if episodes <= 0:
        return []

    size_hint = getattr(env_class, "SIZE", 9)
    _ensure_shared_backend(
        agent_x,
        agent_o,
        capacity=_estimate_backend_capacity(episodes),
        max_actions=size_hint * size_hint,
    )

    if max_workers:
        worker_count = max(1, min(max_workers, episodes))
    else:
        cpu_guess = max(1, (os.cpu_count() or 2) - 1)
        worker_count = max(1, min(cpu_guess, episodes))

    if worker_count <= 1:
        histories = _train_agents_sequential(
            env_class(),
            agent_x,
            agent_o,
            episodes,
            memory_stop_threshold_mb=memory_stop_threshold_mb,
            process=process,
            coin_flip_start=coin_flip_start,
            collect_history=collect_history,
            show_progress=show_progress,
        )
        return histories

    schedule_x: List[float] = []
    schedule_o: List[float] = []

    def build_schedule(agent: BaseAgent, out: List[float]) -> None:
        eps = agent.epsilon
        for _ in range(episodes):
            out.append(eps)
            eps = max(agent.min_epsilon, eps * agent.epsilon_decay)

    build_schedule(agent_x, schedule_x)
    build_schedule(agent_o, schedule_o)

    if batch_size is not None and batch_size > 0:
        chunk_size = batch_size
    else:
        chunk_size = max(1, episodes // (worker_count * 6))
    chunk_size = max(4, min(chunk_size, 32))

    chunks: List[tuple[int, int]] = []
    start = 0
    while start < episodes:
        end = min(start + chunk_size, episodes)
        chunks.append((start, end))
        start = end

    histories: List[List[Dict[str, Any]]] = []
    completed = 0
    progress = tqdm(total=episodes, ncols=80, desc="Training (parallel)", disable=not show_progress)

    mp_ctx = get_context("spawn")

    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=mp_ctx,
            initializer=_init_process_worker,
            initargs=(env_class, agent_x, agent_o, coin_flip_start, collect_history),
        ) as executor:
            futures = []
            for start_idx, end_idx in chunks:
                slice_x = schedule_x[start_idx:end_idx]
                slice_o = schedule_o[start_idx:end_idx]
                futures.append(executor.submit(_process_worker_chunk, slice_x, slice_o))

            for future in as_completed(futures):
                worker_histories, episodes_done = future.result()
                completed += episodes_done
                progress.update(episodes_done)
                if collect_history and worker_histories:
                    histories.extend(worker_histories)
                if memory_limit_hit(process, memory_stop_threshold_mb):
                    for pending in futures:
                        if not pending.done():
                            pending.cancel()
                    break
    finally:
        progress.close()

    def advance_epsilon(start_eps: float, decay: float, minimum: float, steps: int) -> float:
        eps = start_eps
        for _ in range(steps):
            eps = max(minimum, eps * decay)
        return eps

    agent_x.epsilon = advance_epsilon(agent_x.epsilon, agent_x.epsilon_decay, agent_x.min_epsilon, completed)
    agent_o.epsilon = advance_epsilon(agent_o.epsilon, agent_o.epsilon_decay, agent_o.min_epsilon, completed)

    return histories


def _train_agents_sequential(
    env: GameEnv,
    agent_x: BaseAgent,
    agent_o: BaseAgent,
    episodes: int,
    *,
    memory_stop_threshold_mb: Optional[int],
    process,
    coin_flip_start: bool,
    collect_history: bool,
    show_progress: bool,
) -> List[List[Dict[str, Any]]]:
    histories: List[List[Dict[str, Any]]] = []
    episode = 0
    with tqdm(total=episodes, ncols=80, desc="Training", disable=not show_progress) as progress_bar:
        while episode < episodes:
            if memory_limit_hit(process, memory_stop_threshold_mb):
                print("Memory threshold reached; stopping early.")
                break
            try:
                state_history, winner, _ = run_episode(env, agent_x, agent_o, coin_flip_start=coin_flip_start)
                agent_x.learn_result(winner, state_history)
                agent_o.learn_result(winner, state_history)

                if collect_history:
                    histories.append(env.history.copy())

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
            except Exception as exc:
                print(f"Error on episode {episode + 1}: {exc}")
                if env.history and collect_history:
                    histories.append(env.history.copy())
                raise
    return histories


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
    batch_size: Optional[int] = None,
):
    process = get_process() if memory_stop_threshold_mb else None
    all_histories: List[List[Dict[str, Any]]] = []
    collect_history = bool(history_path)

    if parallel:
        all_histories = _train_agents_parallel(
            env_class=type(env),
            agent_x=agent_x,
            agent_o=agent_o,
            episodes=episodes,
            coin_flip_start=coin_flip_start,
            max_workers=max_workers,
            show_progress=show_progress,
            collect_history=collect_history,
            memory_stop_threshold_mb=memory_stop_threshold_mb,
            process=process,
            batch_size=batch_size,
        )
    else:
        all_histories = _train_agents_sequential(
            env,
            agent_x,
            agent_o,
            episodes,
            memory_stop_threshold_mb=memory_stop_threshold_mb,
            process=process,
            coin_flip_start=coin_flip_start,
            collect_history=collect_history,
            show_progress=show_progress,
        )

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
