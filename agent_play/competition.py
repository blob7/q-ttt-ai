from tqdm import tqdm

from agent_play.episode_runner import run_episode
from visualizations.dashboard import GameDashboard


def compete_bots(env, bot1, bot2, episodes, coin_flip_start, *, show_progress=True, visualize=True):
    """Run a competition between two bots for a specified number of episodes."""
    pbar = None
    dashboard = None

    if show_progress:
        pbar = tqdm(total=episodes, desc="Competing Bots", ncols=100)
    if visualize:
        dashboard = GameDashboard(bot1.name, bot2.name)


    for episode in range(episodes):
        _, _, stats = run_episode(env, bot1, bot2, coin_flip_start=coin_flip_start, record_stats=visualize)
        if pbar:
            pbar.update(1)
        if dashboard:
            dashboard.update(stats)