from agents.neural.utils import *
from agents.neural import ObservationalAgent
from agents.neural.obs import Scaler
from agents.neural.obs import *


class NeuralAgent(ObservationalAgent):
    def __init__(self, player: str, env_cfg, obs_scaler: Scaler, model, device, config) -> None:
        super(NeuralAgent, self).__init__(player, env_cfg)

        self.model = model
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

        self.config = config
        self.obs_scaler = obs_scaler

    def create_policy(self, step, obs, remainingOverageTime, transposed=False) -> (np.array, np.array):
        obs_normed = self.obs_scaler.normalize(np.expand_dims(self.obs, 0))

        # generate model policy
        with torch.no_grad():
            inp = torch.tensor(obs_normed).float()
            inp = inp.to(self.device)
            output = self.model(inp)
        move_policy = unload(output.move_policy.squeeze(0))
        sap_policy = unload(output.sap_policy.squeeze(0))

        if transposed:
            move_policy = transpose_policy(move_policy)
            sap_policy = transpose_policy(sap_policy)

        return move_policy, sap_policy
