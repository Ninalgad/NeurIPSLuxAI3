import numpy as np
import torch

from agents.neural.utils import *
from agents.neural.model import UNetModel
from agents.neural import ObservationalAgent


class NeuralAgent(ObservationalAgent):
    def __init__(self, player: str, env_cfg) -> None:
        super(NeuralAgent, self).__init__(player, env_cfg)

        self.model = UNetModel(7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.config = {
            'epsilon': 0.1,
            'hist_size': 3*8,  # 3 frames are added to history each turn
        }

        self.discovered_relic_map = np.zeros((24, 24), dtype='int8')
        self.hist = np.zeros((self.config['hist_size'], 24, 24), dtype='int8')

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.update_internal_state(obs)

        # generate model policy
        with torch.no_grad():
            inp = torch.tensor(self.obs).float().unsqueeze(0).to(self.device)
            output = self.model(inp)
        move_policy = unload(output.move_policy.squeeze(0))
        sap_policy = unload(output.sap_policy.squeeze(0))

        return self._act(obs, move_policy, sap_policy)
