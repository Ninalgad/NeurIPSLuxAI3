import numpy as np
from agents import Agent
from agents.neural.utils import *
from agents.neural import ObservationalAgent


class ObservationalAgentWrapper(ObservationalAgent):

    def __init__(self, base_agent: Agent) -> None:
        super(ObservationalAgentWrapper, self).__init__(base_agent.player, base_agent.env_cfg)
        self.base_agent = base_agent

    def set_env_config(self, env_config):
        self.base_agent.set_env_config(env_config)
        self.env_cfg = env_config

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self.update_internal_state(obs)

        move_policy = np.zeros((5, 24, 24), 'float32') + 1e-4
        sap_policy = np.zeros((2, 24, 24), 'float32') + 1e-4
        unit_positions = obs["units"]["position"][self.team_id]  # shape (max_units, 2)

        actions = self.base_agent.act(step, obs, remainingOverageTime)

        for (m, dx, dy), (x, y) in zip(actions, unit_positions):
            move_policy[m, x, y] = 1.0
            sap_policy[1, x + dx, y + dy] = 1.0

        return self._act(obs, move_policy, sap_policy)