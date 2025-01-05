import numpy as np
from agents.neural.utils import clip_int8


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
        # m = np.maximum(m, transpose_mat(m))  # map features are symmetric
        map_frame[i] = m
    map_frame += 1  # add 1 to make compression easier, since most (hidden) tiles are -1

    # unit energy feature (position is implied)
    unit_frame = np.zeros((1, 24, 24), 'int8')
    for idx in [player_id, opp_id]:
        for (x, y), e in zip(player_obs['units']['position'][idx],
                             player_obs['units']['energy'][idx]):
            if idx == opp_id:
                e = -e
            if (x != -1) and (y != -1):
                e = mu_law_quantize(e, 400, 127)
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
