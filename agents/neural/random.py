import numpy as np

from agents.neural.utils import *
from agents.neural import ObservationalAgent


class RandomAgent(ObservationalAgent):

    def __init__(self, player: str, env_cfg) -> None:
        super(RandomAgent, self).__init__(player, env_cfg)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.update_internal_state(obs)

        move_policy = np.random.uniform(size=(5, 24, 24))
        sap_policy = np.random.uniform(size=(2, 24, 24))

        return self._act(obs, move_policy, sap_policy)


class RushAgent(ObservationalAgent):

    def __init__(self, player: str, env_cfg) -> None:
        super(RushAgent, self).__init__(player, env_cfg)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.update_internal_state(obs)

        move_policy = np.random.uniform(size=(5, 24, 24))
        sap_policy = np.random.uniform(size=(2, 24, 24))

        if self.player == "player_0":
            # rush down and right
            move_policy[2] += 2
            move_policy[3] += 2
        else:
            # rush up and right
            move_policy[1] += 2
            move_policy[4] += 2

        return self._act(obs, move_policy, sap_policy)
