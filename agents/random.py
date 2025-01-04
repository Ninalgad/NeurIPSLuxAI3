import numpy as np

from agents.neural.utils import *
from agents import Agent


class RandomAgent(Agent):

    def __init__(self, player: str, env_cfg) -> None:
        super(RandomAgent, self).__init__(player, env_cfg)

    def act(self, step: int, obs, remainingOverageTime: int = 60):

        action_set = list(range(5))

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in range(self.env_cfg["max_units"]):
            actions[unit_id][0] = np.random.choice(action_set)

        return actions


class RushAgent(Agent):

    def __init__(self, player: str, env_cfg) -> None:
        super(RushAgent, self).__init__(player, env_cfg)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        if self.player == "player_0":
            # rush down and right
            action_set = [2, 3]
        else:
            # rush up and right
            action_set = [1, 4]

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for unit_id in range(self.env_cfg["max_units"]):
            actions[unit_id][0] = np.random.choice(action_set)

        return actions
