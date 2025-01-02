import abc


class Agent(metaclass=abc.ABCMeta):
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.team_id = 0 if self.player == "player_0" else 1
        self.opp_team_id = 1 if self.team_id == 0 else 0
        # np.random.seed(0)
        self.env_cfg = env_cfg

        self.obs = None
        self.action = None

        self.move_policy_mask = None
        self.sap_policy_mask = None

    def set_env_config(self, env_config):
        self.env_cfg = env_config

    @abc.abstractmethod
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        pass
