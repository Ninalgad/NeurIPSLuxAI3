from agents import Agent
import numpy as np
import abc

from agents.neural.utils import *
from agents.neural.obs import create_obs_frame


class ObservationalAgent(Agent, metaclass=abc.ABCMeta):

    def __init__(self, player: str, env_cfg) -> None:
        super(ObservationalAgent, self).__init__(player, env_cfg)

        self.config = {
            'epsilon': 0.01,
            'hist_size': 3 * 8,  # 3 frames are added to history each turn
        }

        self.discovered_relic_map = np.zeros((1, 24, 24), dtype='int8')
        self.hist = np.zeros((self.config['hist_size'], 24, 24), dtype='int8')

        self.running_board_state = np.zeros((3, 24, 24), dtype='int8')
        self.running_point_gains = np.zeros((1, 24, 24), dtype='int8') - 127
        self.last_team_points = 0

    def _get_vector_state(self, player_obs):
        t, o = self.team_id, self.opp_team_id
        v = [
            int(player_obs['match_steps'] / 102), player_obs['match_steps'] % 102,
            player_obs['team_points'][t] - 127,
            player_obs['team_points'][t] - player_obs['team_points'][o],
            max(-1, player_obs['team_points'][t] - self.last_team_points)
        ]
        return [clip_int8(x) for x in v]

    def _update_internal_state(self, obs):
        # update discovered relic positions
        for pos, unmask in zip(obs["relic_nodes"], obs["relic_nodes_mask"]):
            if unmask:
                for (x, y) in [pos, transpose(pos)]:
                    self.discovered_relic_map[0, x, y] = 1

        # update map state
        for i, k in enumerate(['energy', 'tile_type']):
            m = obs['map_features'][k]
            mask = m > -1
            self.running_board_state[i][mask] = m[mask] + 1
        self.running_board_state[-1][mask] = 127  # records time since last seen
        self.running_board_state[-1] = clip_int8(self.running_board_state[-1] - 1)

        # update point gains map
        point_gain = max(0, obs['team_points'][self.team_id] - self.last_team_points)
        unit_positions = obs["units"]["position"][self.team_id]  # shape (max_units, 2)
        for x, y in unit_positions:
            self.running_point_gains[0, x, y] = clip_int8(point_gain + self.running_point_gains[0, x, y])

        self.obs, self.hist = create_obs_frame(
            obs, self.hist,
            self._get_vector_state(obs),
            [self.discovered_relic_map, self.running_board_state, self.running_point_gains],
            self.team_id, self.opp_team_id
        )
        self.last_team_points = obs['team_points'][self.team_id]

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        self._update_internal_state(obs)

        move_policy, sap_policy = self.create_policy(step, obs, remainingOverageTime)

        assert move_policy.shape == (5, 24, 24), move_policy.shape
        assert sap_policy.shape == (2, 24, 24), sap_policy.shape

        self.move_policy_mask = np.zeros((5, 24, 24), 'int8')
        self.sap_policy_mask = np.zeros((2, 24, 24), 'int8')

        unit_mask = obs["units_mask"][self.team_id]  # shape (max_units, )
        unit_positions = obs["units"]["position"][self.team_id]  # shape (max_units, 2)
        actions = np.zeros((self.env_cfg["max_units"], 3), dtype=int)

        # movement actions
        for unit_id, (unit_pos, unmask) in enumerate(zip(unit_positions, unit_mask)):
            if unmask:
                if np.random.uniform() < self.config['epsilon']:
                    m = np.random.choice(5)
                else:
                    m = np.argmax(move_policy[:, unit_pos[0], unit_pos[1]])
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
                                s = sap_policy[1, target_pos[0], target_pos[1]] + 0.1*np.random.gumbel()
                                if s > best_score:
                                    best_dx = [dx, dy]
                                    best_score = s
                                    best_pos = target_pos
                                self.sap_policy_mask[0, target_pos[0], target_pos[1]] = 1

                    if np.random.uniform() < self.config['epsilon']:
                        s = np.random.choice(2)
                    else:
                        s = np.argmax(sap_policy[:, best_pos[0], best_pos[1]])

                    if s == 1:
                        actions[unit_id][1:] = best_dx

                    self.sap_policy_mask[s, best_pos[0], best_pos[1]] = 1

        # JAX breaks without the following:
        actions_ = np.zeros((self.env_cfg["max_units"], 3), dtype=int)
        for i, act in enumerate(actions):
            actions_[i] = act

        return actions_

    @abc.abstractmethod
    def create_policy(self, step, obs, remainingOverageTime) -> (np.array, np.array):
        """Create the action policies"""
