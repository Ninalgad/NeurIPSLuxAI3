import torch
import numpy as np


class NetworkCacher:
    """An object to share the network between the self-play and training jobs."""

    def __init__(self, max_models):
        self._networks = {}
        self._step = 0
        self.max_models = max_models

    def save_network(self, network: torch.nn.Module):
        self._networks[self._step] = network.state_dict()
        self._step = (self._step + 1) % self.max_models

    def load_network(self, step: int) -> dict:
        return self._networks[step]

    def sample_network(self):
        s = np.random.choice(list(self._networks.keys()))
        return self.load_network(s)

    def save(self, filepath):
        torch.save(self._networks, filepath)

    def load(self, filepath, device):
        self._networks = torch.load(filepath, device)
