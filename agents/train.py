from typing import Optional
from pathlib import Path

from game.environment import GameEnv
from game.board import Winner

from agents.base_agent import BaseAgent

def train_agents(env: GameEnv, agent_x: BaseAgent, agent_o: BaseAgent, episodes: int = 1000, download_path: Optional[str] = None):
    all_histories = []

    for episode in range(episodes):
        try:
            state = env.reset()
            done = False
            current_agent = agent_x

            while not done:
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
                    other_agent = agent_o if current_agent == agent_x else agent_x
                    other_agent.learn(state, None, new_state, done, winner)

                # Swap turns
                state = new_state
                current_agent = agent_o if current_agent == agent_x else agent_x

            all_histories.append(env.history.copy())
            if (episode + 1) % 1000 == 0:
                print(f"Completed episode {episode + 1}/{episodes}")

        except Exception as e:
            print(f"Error on episode {episode + 1}: {e}")
            if env.history:
                all_histories.append(env.history.copy())
            if download_path:
                print("Saving collected match histories before exiting...")
                _download_training_data(all_histories, Path(download_path))
            raise

    if download_path:
        _download_training_data(all_histories, Path(download_path))




def _download_training_data(all_histories: list[dict], download_path: Path):
    import zipfile
    import json

    download_path = Path(download_path)
    download_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    with zipfile.ZipFile(download_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for i, history in enumerate(all_histories):
            # Convert history to JSON string
            json_bytes = json.dumps([
                {"player": entry.get("player"), "move": list(entry.get("move"))}
                for entry in history
            ], ensure_ascii=False, indent=2).encode("utf-8")

            # Save as episode_i.json inside the ZIP
            zipf.writestr(f"episode_{i+1}.json", json_bytes)