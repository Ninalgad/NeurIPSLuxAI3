import scipy
import numpy as np
from luxai_s3.wrappers import LuxAIS3GymEnv, RecordEpisode
from utils import *


def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def player_values(trajectory, player_id, opp_id):
    r = np.array([x.reward for x in trajectory[player_id]])
    o = np.array([x.reward for x in trajectory[opp_id]])
    p = np.array([x.points for x in trajectory[player_id]])

    r = r[1:] - r[:-1]
    o = o[1:] - o[:-1]
    p = p[1:] - p[:-1]

    r = r - o
    r *= 5
    p = np.clip(p, 0, 1)

    r = discounted_cumulative_sums(r, 0.9998)
    p = discounted_cumulative_sums(p, 0.7777)

    v = r + p

    # v = v * ((1 + np.arange(1, len(r) + 1))/len(r)) ** 0.99 # weigh team wins value at the end more
    v /= 9
    v = np.pad(v, (1, 0))

    return v


def finish_trajectory(trajectory):
    values0 = player_values(trajectory, 'player_0', 'player_1')
    values1 = player_values(trajectory, 'player_1', 'player_0')

    trajectory['player_0'] = [s._replace(value=v)
                              for v, s in zip(values0, trajectory['player_0'])]
    trajectory['player_1'] = [s._replace(value=v)
                              for v, s in zip(values1, trajectory['player_1'])]
    return trajectory


def create_action_maps(actions, player_obs, team_id):
    move_policy_mask = np.zeros((5, 24, 24), 'int8')
    sap_policy_mask = np.zeros((2, 24, 24), 'int8')

    unit_mask = np.array(player_obs["units_mask"][team_id])  # shape (max_units, )
    unit_positions = np.array(player_obs["units"]["position"][team_id])  # shape (max_units, 2)

    for action, unmask, pos in zip(actions, unit_mask, unit_positions):
        move, dx, dy = action
        x, y = pos
        move_policy_mask[move, x, y] = 1

        if (dx == 0) and (dy == 0):
            sap_policy_mask[0, x]


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

        for i, agent in enumerate([player_0, player_1]):
            s = State(0,
                      obs['player_0']["team_points"][i],
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
