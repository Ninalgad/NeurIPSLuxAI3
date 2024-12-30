import numpy as np
import torch

from agents import Agent
from agents.neural.utils import *
from agents.neural.model import UNetModel


def get_policy(policy_map):
    p = unload(policy_map.squeeze(0))
    p = np.argmax(p, axis=0)
    return p


class NeuralAgent(Agent):
    def __init__(self, player: str, env_cfg) -> None:
        super(NeuralAgent, self).__init__(player, env_cfg)

        self.model = UNetModel(7)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.epsilon = 0

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """implement this function to decide what actions to send to each available unit.

        step is the current timestep number of the game starting from 0 going up to max_steps_in_match * match_count_per_episode - 1.
        """
        self.obs = create_obs_frame(obs, self.team_id, self.opp_team_id)

        unit_mask = np.array(obs["units_mask"][self.team_id])  # shape (max_units, )
        unit_positions = np.array(obs["units"]["position"][self.team_id])  # shape (max_units, 2)

        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        with torch.no_grad():
            inp = torch.tensor(self.obs).float().unsqueeze(0).to(self.device)
            output = self.model(inp)

        move_policy = unload(output.move_policy.squeeze(0))
        sap_policy = unload(output.sap_policy.squeeze(0))

        assert move_policy.shape == (5, 24, 24), move_policy.shape
        assert sap_policy.shape == (2, 24, 24), sap_policy.shape

        self.move_policy_mask = np.zeros((5, 24, 24), 'int8')
        self.sap_policy_mask = np.zeros((2, 24, 24), 'int8')

        # movement actions
        for unit_id, (unit_pos, unmask) in enumerate(zip(unit_positions, unit_mask)):
            if unmask:
                if np.random.uniform() < self.epsilon:
                    m = np.random.choice(5)
                else:
                    m = move_policy[:, unit_pos[0], unit_pos[1]]
                    m = np.random.choice(5, p=m/m.sum())
                actions[unit_id][0] = m
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
