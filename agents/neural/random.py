import numpy as np

from agents import Agent
from agents.neural.utils import *


class RandomAgent(Agent):

    def __init__(self, player: str, env_cfg) -> None:
        super(RandomAgent, self).__init__(player, env_cfg)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.obs = create_obs_frame(obs, self.team_id, self.opp_team_id)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        move_policy = np.random.randint(0, 5, size=(24, 24), dtype='int8')
        sap_policy = np.random.randint(0, 2, size=(24, 24), dtype='int8')

        self.move_policy_mask = np.zeros((5, 24, 24), 'int8')
        self.sap_policy_mask = np.zeros((2, 24, 24), 'int8')

        # movement actions
        for unit_id, (unit_pos, unmask) in enumerate(zip(unit_positions, unit_mask)):
            if unmask:
                m = move_policy[unit_pos[0], unit_pos[1]]
                actions[unit_id][0] = m
                self.move_policy_mask[m, unit_pos[0], unit_pos[1]] = 1

        # sap actions
        if np.any(obs["units_mask"][self.opp_team_id]):
            sap_range = self.env_cfg['unit_sap_range']
            target_range = np.arange(-sap_range + 1, sap_range)
            for unit_id, (unit_pos, unmask) in enumerate(zip(unit_positions, unit_mask)):
                if unmask:
                    for dx in target_range:
                        for dy in target_range:
                            target_pos = unit_pos[0] + dx, unit_pos[1] + dy

                            if is_valid_pos(target_pos) and (get_distance(target_pos, unit_pos) <= sap_range):
                                s = sap_policy[target_pos[0], target_pos[1]]
                                if s == 1:
                                    actions[unit_id][1:] = [dx, dy]
                                    self.sap_policy_mask[s, target_pos[0], target_pos[1]] = 1

        return actions
