from tqdm import tqdm

from agent_play.episode_runner import run_episode
from visualizations.dashboard import GameDashboard


def compete_bots(env, bot1, bot2, episodes, coin_flip_start, *, show_progress=True, visualize=True, bot1_name=None, bot2_name=None):
    """Run a competition between two bots for a specified number of episodes."""
    pbar = None
    dashboard = None

    if show_progress:
        pbar = tqdm(total=episodes, desc="Competing Bots", ncols=100)
    if visualize:
        dashboard = GameDashboard(bot1_name or bot1.name, bot2_name or bot2.name)


    for episode in range(episodes):
        _, _, stats = run_episode(env, bot1, bot2, coin_flip_start=coin_flip_start, record_stats=visualize)
        if pbar:
            pbar.update(1)
        if dashboard and stats:
            if bot1_name and stats['winning_player'] == bot1.name:
                stats['winning_player'] = bot1_name
            if bot2_name and stats['winning_player'] == bot2.name:
                stats['winning_player'] = bot2_name

            dashboard.update(stats)
    if dashboard:
        dashboard.finalize()