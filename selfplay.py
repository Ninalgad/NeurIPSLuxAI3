import scipy
import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from utils import *


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def finish_trajectory(trajectory):
    r0 = np.array([x.reward for x in trajectory['player_0']])
    r1 = np.array([x.reward for x in trajectory['player_1']])

    r = r1 - r0
    r = r * np.arange(1, len(r) + 1)
    r = r[:-1] - r[1:]
    r = discounted_cumulative_sums(r, 0.99).astype('float32')
    r = r / np.abs(r).max()

    trajectory['player_0'] = [s._replace(value=-v)
                              for v, s in zip(r, trajectory['player_0'])]
    trajectory['player_1'] = [s._replace(value=v)
                              for v, s in zip(r, trajectory['player_1'])]
    return trajectory


def run_selfplay(player_0, player_1, seed=0, replay_save_dir="",
                 display_episode=False):

    env = LuxAIS3GymEnv(numpy_output=True)
    if display_episode:
        env = RecordEpisode(
            env, save_on_close=True, save_on_reset=True, save_dir=replay_save_dir
        )  # used for render_episode
    obs, info = env.reset(seed=seed)

    env_cfg = info["params"]  # only contains observable game parameters
    player_0.env_cfg = env_cfg
    player_1.env_cfg = env_cfg

    traj = {agent.player: [] for agent in [player_0, player_1]}

    # main game loop
    game_done = False
    step = 0
    while not game_done:
        actions = dict()
        for agent in [player_0, player_1]:
            actions[agent.player] = agent.act(step=step, obs=obs[agent.player])

        obs, reward, terminated, truncated, info = env.step(actions)

        for agent in [player_0, player_1]:
            traj[agent.player].append(State(0, reward[agent.player].item(), agent.obs,
                                            agent.move_policy_mask, agent.sap_policy_mask))

        dones = {k: terminated[k] | truncated[k] for k in terminated}
        if dones["player_0"] or dones["player_1"]:
            game_done = True
        step += 1

    if display_episode:
        render_episode(env)
    env.close()

    return finish_trajectory(traj)
