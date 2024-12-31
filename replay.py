import numpy as np
import os
from scipy import sparse

from utils import *


class ReplayBuffer(object):

    def __init__(self, window_size, batch_size, obs_dim):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer = None
        self.obs_dim = obs_dim

    def get_size(self):
        return len(self.buffer['value'])

    def save_game(self, game: dict):
        for states in game.values():
            for s in states:
                self.append_state(s)

    def append_state(self, s: State):
        assert s.obs.shape[0] == self.obs_dim, f"Expected obs to have dim {self.obs_dim} got {s.obs.shape[0]}"
        s = {
            'value': np.array(s.value, 'float32'),
            'obs': s.obs,
            'move_action': s.move_action,
            'sap_action': s.sap_action
            }
        s = {k: np.expand_dims(v, 0) for (k, v) in s.items()}

        if self.buffer is None:
            self.buffer = s
        else:
            for (k, v) in self.buffer.items():
                self.buffer[k] = np.append(v, s[k], axis=0)[-self.window_size:]

    def sample_batch(self) -> dict:
        batch = None
        for _ in range(self.batch_size):
            s = self.sample()
            if batch is None:
                batch = s
            else:
                for (k, v) in s.items():
                    batch[k] = np.append(batch[k], v, axis=0)
        return batch

    def sample(self) -> dict:
        i = np.random.choice(self.get_size())
        item = {k: np.expand_dims(v[i], 0) for (k, v) in self.buffer.items()}
        return item

    def save(self, directory):
        np.save(os.path.join(directory, 'value.npy'), self.buffer['value'])

        for k in ['obs', 'move_action', 'sap_action']:
            v = self.buffer[k].copy()
            v = v.reshape((-1, 24))
            v = sparse.csr_matrix(v)
            sparse.save_npz(os.path.join(directory, f'{k}.npz'), v)

    def load(self, directory):
        buffer = dict()
        buffer['value'] = np.load(os.path.join(directory, 'value.npy')).astype('float32')

        x = sparse.load_npz(os.path.join(directory, 'obs.npz')).toarray().reshape((-1, self.obs_dim, 24, 24))
        buffer['obs'] = x.astype('int32')

        x = sparse.load_npz(os.path.join(directory, 'move_action.npz')).toarray().reshape((-1, 5, 24, 24))
        buffer['move_action'] = x.astype('int8')

        x = sparse.load_npz(os.path.join(directory, 'sap_action.npz')).toarray().reshape((-1, 2, 24, 24))
        buffer['sap_action'] = x.astype('int8')

        self.buffer = buffer
