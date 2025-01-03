import numpy as np
import os
import gc
from scipy import sparse

from utils import *


class ReplayBuffer(object):

    def __init__(self, window_size, batch_size, obs_dim):
        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer = None
        self.obs_dim = obs_dim

    def get_size(self):
        return 0 if (self.buffer is None) else len(self.buffer['value'])

    def save_game(self, game: dict):
        for states in game.values():
            self.append_states(states)

    def append_states(self, states: list[State]):
        buffer_update = {
            'value': [],
            'obs': [],
            'move_action': [],
            'sap_action': []
        }
        for s in states:
            buffer_update['value'].append(s.value)
            buffer_update['obs'].append(s.obs)
            buffer_update['move_action'].append(s.move_action)
            buffer_update['sap_action'].append(s.sap_action)

        buffer_update['value'] = np.array(buffer_update['value'], "float32")
        for k in ['obs', 'move_action', 'sap_action']:
            buffer_update[k] = np.array(buffer_update[k], "int8")

        msg = f"Expected obs to have dim {(-1, self.obs_dim, 24, 24)} got {buffer_update['obs'].shape}"
        assert buffer_update['obs'].shape[1] == self.obs_dim, msg
        if self.buffer is None:
            self.buffer = buffer_update
        else:
            for (k, v) in self.buffer.items():
                self.buffer[k] = np.append(v, buffer_update[k], axis=0)[-self.window_size:]

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
        np.save(os.path.join(directory, 'value.npy'), self.buffer['value'].astype('float32'))
        gc.collect()

        for k in ['obs', 'move_action', 'sap_action']:
            v = self.buffer[k].copy().astype('int32')
            v = v.reshape((-1, 24))
            v = sparse.csr_matrix(v)
            sparse.save_npz(os.path.join(directory, f'{k}.npz'), v)
            del v
            gc.collect()

    def load(self, directory):
        buffer = dict()
        buffer['value'] = np.load(os.path.join(directory, 'value.npy')).astype('float32')

        x = sparse.load_npz(os.path.join(directory, 'obs.npz')).toarray().reshape((-1, self.obs_dim, 24, 24))
        buffer['obs'] = x.astype('int8')

        x = sparse.load_npz(os.path.join(directory, 'move_action.npz')).toarray().reshape((-1, 5, 24, 24))
        buffer['move_action'] = x.astype('int8')

        x = sparse.load_npz(os.path.join(directory, 'sap_action.npz')).toarray().reshape((-1, 2, 24, 24))
        buffer['sap_action'] = x.astype('int8')

        self.buffer = buffer
