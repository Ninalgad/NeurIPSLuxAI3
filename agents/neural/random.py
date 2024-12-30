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

        move_policy = np.random.uniform(size=(5, 24, 24))
        sap_policy = np.random.uniform(size=(2, 24, 24))

        self.move_policy_mask = np.zeros((5, 24, 24), 'int8')
        self.sap_policy_mask = np.zeros((2, 24, 24), 'int8')

        # movement actions
        for unit_id, (unit_pos, unmask) in enumerate(zip(unit_positions, unit_mask)):
            if unmask:
                m = move_policy[:, unit_pos[0], unit_pos[1]]
                m = np.random.choice(5, p=m/m.sum())
                self.move_policy_mask[m, unit_pos[0], unit_pos[1]] = 1

        # sap actions
        if np.any(obs["units_mask"][self.opp_team_id]):
            sap_range = self.env_cfg['unit_sap_range']
            target_range = np.arange(-sap_range + 1, sap_range)
            for unit_id, (unit_pos, unmask) in enumerate(zip(unit_positions, unit_mask)):
                if unmask:
                    best_pos, best_dx, best_score = [0, 0], [0, 0], 0
                    for dx in target_range:
                        for dy in target_range:
                            target_pos = unit_pos[0] + dx, unit_pos[1] + dy

                            if is_valid_pos(target_pos) and (get_distance(target_pos, unit_pos) <= sap_range):
                                s = sap_policy[1, target_pos[0], target_pos[1]]
                                if s > best_score:
                                    best_dx = [dx, dy]
                                    best_score = s
                                    best_pos = target_pos
                                self.sap_policy_mask[0, target_pos[0], target_pos[1]] = 1

                    actions[unit_id][1:] = best_dx

                    self.sap_policy_mask[1, best_pos[0], best_pos[1]] = 1

        return actions
