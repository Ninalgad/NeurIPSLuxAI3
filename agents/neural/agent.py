from agents.neural.utils import *
from agents.neural import ObservationalAgent


class NeuralAgent(ObservationalAgent):
    def __init__(self, player: str, env_cfg, model, device, config) -> None:
        super(NeuralAgent, self).__init__(player, env_cfg)

        self.model = model
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()

        self.config = config

    def create_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # generate model policy
        with torch.no_grad():
            inp = torch.tensor(self.obs).float().unsqueeze(0).to(self.device)
            output = self.model(inp)
        move_policy = unload(output.move_policy.squeeze(0))
        sap_policy = unload(output.sap_policy.squeeze(0))

        return move_policy, sap_policy
