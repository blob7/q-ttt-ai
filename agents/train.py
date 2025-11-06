from typing import Optional
from pathlib import Path
import random

from game.environment import GameEnv

from agents.base_agent import BaseAgent

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil may not be available
    psutil = None


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
    all_histories = []
    stop_due_to_memory = False

    process = psutil.Process() if (psutil and memory_stop_threshold_mb) else None
    if memory_stop_threshold_mb and process is None:
        print("psutil not available; memory_stop_threshold_mb ignored.")
        memory_stop_threshold_mb = None

    progress_segments = 100 if show_progress else 0
    last_progress_segment = -1
    progress_bar_active = False

    episode = 0
    stop_due_to_interrupt = False

    while episode < episodes:
        if _memory_limit_hit(process, memory_stop_threshold_mb):
            progress_bar_active = _ensure_progress_newline(progress_bar_active)
            print("Memory threshold reached before starting new episode; stopping training early.")
            stop_due_to_memory = True
            break
        try:
            state = env.reset()
            done = False
            first_agent = agent_x
            second_agent = agent_o

            if coin_flip_start:
                if random.random() < 0.5:
                    first_agent, second_agent = second_agent, first_agent
                desired_player = getattr(first_agent, "role", None)
                if desired_player in (1, -1):
                    env.game.current_player = desired_player
                    state = env.get_state()

            current_agent = first_agent

            while not done:
                if _memory_limit_hit(process, memory_stop_threshold_mb):
                    progress_bar_active = _ensure_progress_newline(progress_bar_active)
                    print("Memory threshold reached mid-episode; stopping training early.")
                    stop_due_to_memory = True
                    break
                valid_moves = env.get_valid_moves()
                if not valid_moves:
                    # Edge case: no moves left
                    done = True
                    winner = 0
                    current_agent.learn(state, None, state, done, winner)
                    break

                action = current_agent.choose_action(state, valid_moves)
                new_state, done, winner = env.step(action)

                # Current agent always learns
                current_agent.learn(state, action, new_state, done, winner)

                # If the game ended, inform the other agent too
                if done:
                    other_agent = second_agent if current_agent is first_agent else first_agent
                    other_agent.learn(state, None, new_state, done, winner)

                # Swap turns
                state = new_state
                current_agent = second_agent if current_agent is first_agent else first_agent

            if stop_due_to_memory:
                break

            if history_path:
                all_histories.append(env.history.copy())

            if show_progress and episodes > 0:
                last_progress_segment, printed = _update_progress_bar(
                    completed=episode + 1,
                    total=episodes,
                    last_segment=last_progress_segment,
                    segments=progress_segments,
                )
                if printed:
                    progress_bar_active = True

            episode += 1

        except KeyboardInterrupt:
            progress_bar_active = _ensure_progress_newline(progress_bar_active)
            try:
                choice = input("Training interrupted. Cancel training? [y/N]: ").strip().lower()
            except EOFError:
                choice = "y"
            if choice in ("y", "yes"):
                print("Training cancelled by user.")
                stop_due_to_interrupt = True
                break
            print("Resuming training...")
            last_progress_segment = -1
            progress_bar_active = False
            continue

        except Exception as e:
            progress_bar_active = _ensure_progress_newline(progress_bar_active)
            print(f"Error on episode {episode + 1}: {e}")
            if env.history:
                all_histories.append(env.history.copy())
            if history_path:
                print("Saving collected match histories before exiting...")
                _download_training_data(all_histories, Path(history_path))
            raise

    if show_progress and progress_bar_active:
        print()

    if (stop_due_to_memory or stop_due_to_interrupt) and env.history and history_path:
        all_histories.append(env.history.copy())

    if history_path and all_histories:
        _download_training_data(all_histories, Path(history_path))

    _save_agent_snapshot(agent_x, agent_x_save_path)
    _save_agent_snapshot(agent_o, agent_o_save_path)




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


def _memory_limit_hit(process, threshold_mb: Optional[int]) -> bool:
    if not threshold_mb or not process:
        return False
    try:
        rss_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        return False
    return rss_mb >= threshold_mb


def _save_agent_snapshot(agent: BaseAgent, path_hint: Optional[str]) -> None:
    if agent is None or not path_hint:
        return
    target_path = Path(path_hint)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(target_path))


def _update_progress_bar(completed: int, total: int, last_segment: int, segments: int) -> tuple[int, bool]:
    if total <= 0:
        return last_segment, False
    segment = min(segments, int((completed * segments) / total))
    if segment == last_segment:
        return last_segment, False
    percent = int((completed * 100) / total)
    percent = min(100, percent)
    bar = "#" * segment + "-" * (segments - segment)
    print(f"\rTraining [{bar}] {completed}/{total} ({percent:3d}%)", end="", flush=True)
    return segment, True


def _ensure_progress_newline(progress_active: bool) -> bool:
    if progress_active:
        print()
        return False
    return progress_active