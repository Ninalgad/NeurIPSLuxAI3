import numpy as np
from agents.neural.utils import *


def mu_law_quantize(x, input_max, output_max, mu=8):
    # µ-law companding transformation
    x = np.array(x, 'float32') / input_max
    x = np.sign(x) * np.log1p(mu*np.abs(x)) / np.log1p(mu)
    x = output_max * np.rint(x)
    return x


def create_obs_frame(player_obs, hist_frames, vector_state, additional_frames, player_id, opp_id):

    # map features
    map_frame = np.zeros((2, 24, 24), 'int8')
    for i, k in enumerate(['energy', 'tile_type']):
        m = player_obs['map_features'][k]
        map_frame[i] = clip_int8(m)
    map_frame += 1  # add 1 to make compression easier, since most (hidden) tiles are -1

    # unit energy feature (position is implied)
    unit_frame = np.zeros((1, 24, 24), 'int8')
    for idx in [player_id, opp_id]:
        for (x, y), e in zip(player_obs['units']['position'][idx],
                             player_obs['units']['energy'][idx]):
            if idx == opp_id:
                e = -e
            if (x != -1) and (y != -1):
                e = mu_law_quantize(e, 400, 127)  # [0, 400] -> [0, 127]
                e = clip_int8(e).astype('int8')
                unit_frame[0, x, y] = e

    # update hist
    hist_frames = np.roll(hist_frames, 3, axis=0)
    hist_frames[:3] = np.concatenate([unit_frame, map_frame], axis=0, dtype='int8')

    # vector information frames
    v_frames = np.zeros((len(vector_state), 24, 24), 'int8')
    vector_state = np.array(vector_state, 'int8')[:, None, None]
    v_frames = v_frames + vector_state

    frames = np.concatenate(additional_frames + [v_frames, hist_frames], axis=0, dtype='int8')
    return frames, hist_frames


def transpose_pos(pos):
    if pos[0] == -1:
        return -1, -1
    return 23 - pos[1], 23 - pos[0]


def transpose_unit_pos(units):
    t0 = [transpose_pos(p) for p in units[0]]
    t1 = [transpose_pos(p) for p in units[1]]
    return np.array([t0, t1], units.dtype)


def transpose_action(act):
    # (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    action_transpose = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2}
    m, dx, dy = act
    return [action_transpose[m], -dx, -dy]


def transpose_relic_nodes(a):
    return np.array([transpose_pos(x) for x in a], a.dtype)


def transpose_grid(a):
    return a.copy()[::-1, ::-1].T


def transpose_obs(obs):
    obs_t = {}
    for (k, v) in obs.items():
        if k == "units":
            v = {'position': transpose_unit_pos(v['position']), 'energy': v['energy']}
        elif k == "sensor_mask":
            v = transpose_grid(v)
        elif k == "map_features":
            v = {name: transpose_grid(feat) for (name, feat) in v.items()}
        elif k == "relic_nodes":
            v = transpose_relic_nodes(v)
        else:
            v = v.copy()

        obs_t[k] = v
    return obs_t


def transpose_policy(policy):
    if policy is not None:
        # translate positions
        policy = np.transpose(policy.copy(), (0, 2, 1))[:, ::-1, ::-1]

        if policy.shape[0] == 5:
            # translate moves
            move_policy = policy.copy()
            move_policy[1] = policy[4]
            move_policy[4] = policy[1]
            move_policy[2] = policy[3]
            move_policy[3] = policy[2]
            policy = move_policy
        return policy


class Scaler:
    def __init__(self, axis=1):
        """MinMax Scaler"""
        self.maximums = -np.inf
        self.minimums = np.inf
        self.axis = axis

    def update(self, x):
        self.maximums = np.maximum(
            self.maximums, np.max(x, axis=self.axis, keepdims=True)).astype('float32')
        self.minimums = np.minimum(
            self.minimums, np.min(x, axis=self.axis, keepdims=True)).astype('float32')

    def normalize(self, x):
        x = x.astype('float32')
        x = (x - self.minimums) / (self.maximums - self.minimums + 1e-10)  # [0, 1]
        return 2*x - 1  # [-1, 1]

    def save(self, filename):
        np.save(filename, {'maximums': self.maximums, 'minimums': self.minimums})

    def load(self, filename):
        params = np.load(filename, allow_pickle=True).item()
        self.maximums = params['maximums']
        self.minimums = params['minimums']
