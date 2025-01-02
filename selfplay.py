import scipy
import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from agents import Agent
from utils import *


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def player_values(trajectory, player_id, opp_id):
    e = np.array([x.energy for x in trajectory[player_id]], 'float32')
    p = np.array([x.points for x in trajectory[player_id]], 'float32')
    d = np.array([x.reward - y.reward for x, y in zip(trajectory[player_id], trajectory[opp_id])], 'float32')

    p = discounted_cumulative_sums(p / 500, 0.7777)
    e = np.maximum(np.pad(e[1:] - e[:-1], (1, 0)), 0)

    v = p + (d / 5) + (1e-3 * e)
    v = np.clip(v, -1, 1)

    return v


def finish_trajectory(trajectory):
    values0 = player_values(trajectory, 'player_0', 'player_1')
    values1 = player_values(trajectory, 'player_1', 'player_0')

    trajectory['player_0'] = [s._replace(value=v)
                              for v, s in zip(values0, trajectory['player_0'])]
    trajectory['player_1'] = [s._replace(value=v)
                              for v, s in zip(values1, trajectory['player_1'])]
    return trajectory


def run_selfplay(player_0: Agent, player_1: Agent, seed: int = 0, replay_save_dir: str = "",
                 display_episode: bool = False):

    env = LuxAIS3GymEnv(numpy_output=True)
    if display_episode:
        env = RecordEpisode(
            env, save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
        )  # used for render_episode
    obs, info = env.reset(seed=seed)

    env_cfg = info["params"]  # only contains observable game parameters
    player_0.set_env_config(env_cfg)
    player_1.set_env_config(env_cfg)

    traj = {agent.player: [] for agent in [player_0, player_1]}

    # main game loop
    game_done = False
    step = 0
    while not game_done:
        actions = dict()
        for agent in [player_0, player_1]:
            actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

        obs, reward, terminated, truncated, info = env.step(actions)

        for i, agent in enumerate([player_0, player_1]):
            s = State(0,
                      obs['player_0']["team_points"][i],
                      obs[agent.player]['units']['energy'][i].sum(),
                      reward[agent.player].item(),
                      agent.obs,
                      agent.move_policy_mask,
                      agent.sap_policy_mask)
            traj[agent.player].append(s)

        dones = {k: terminated[k] | truncated[k] for k in terminated}
        if dones["player_0"] or dones["player_1"]:
            game_done = True
        step += 1

    if display_episode:
        render_episode(env)
    env.close()

    return finish_trajectory(traj)
