import abc
import torch
from replay import ReplayBuffer
from agents.neural.utils import NetworkOutput
from agents.neural.obs import Scaler


class Learner(metaclass=abc.ABCMeta):
    """A learner to update the network weights based."""

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 learning_rate: float):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model
        self.device = device
        self.model.to(device)
        self.steps = 0

    @abc.abstractmethod
    def get_loss(self, outputs: NetworkOutput, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Objective function to optimize"""

    def learn(self, replay_buffer: ReplayBuffer, scaler: Scaler) -> float:
        """Single training step of the learner."""

        self.model.train()
        batch = replay_buffer.sample_batch()
        batch['obs'] = scaler.normalize(batch['obs'])
        batch = {k: torch.tensor(v).float().to(self.device) for (k, v) in batch.items()}

        batch['value'] = batch['value'].unsqueeze(1)

        self.optimizer.zero_grad()
        outputs = self.model(batch['obs'])

        assert outputs.value.shape == batch['value'].shape, (outputs.value.shape, batch['value'].shape)

        loss_ = self.get_loss(outputs, batch)
        loss_.backward()

        self.optimizer.step()
        self.steps += 1

        return loss_.detach().cpu().numpy().item()

    def export(self, filepath, meta=None):
        state = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.steps
        }
        if meta is not None:
            state = {**state, **meta}
        torch.save(state, filepath)

    def import_(self, filepath):
        checkpoint = torch.load(filepath, self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['step']
        return {k: v for (k, v) in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict', 'step']}

    def export_model_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)


class A2CLearner(Learner):
    """Implements the Advantage Actor Critic (A2C) learning algorithm."""

    def get_loss(self, outputs: NetworkOutput, batch: dict[str, torch.tensor]):
        def actor_loss(policy, adv, action):
            a = adv[:, :, None, None, None] * action
            log_policy = torch.log(policy + 1e-5)
            loss = - (log_policy * a)

            return loss.sum() / (action.sum() + 1e-5)

        def critic_loss(prediction, target):
            return torch.nn.L1Loss()(prediction, target)

        target_value = batch['value']
        predicted_value = outputs.value

        loss_ = critic_loss(predicted_value, target_value)

        advantage = predicted_value - target_value
        advantage = advantage.detach()

        loss_ += 0.5 * actor_loss(outputs.move_policy, advantage, batch['move_action'])
        loss_ += 0.5 * actor_loss(outputs.sap_policy, advantage, batch['sap_action'])

        return loss_


class BCLearner(Learner):
    """Implements the behavioral cloning learning algorithm."""

    def get_loss(self, outputs: NetworkOutput, batch: dict[str, torch.tensor]) -> torch.Tensor:
        def critic_loss(prediction, target):
            return torch.nn.L1Loss()(prediction, target)

        def actor_loss(output, target):
            eps = 1e-10
            loss = -torch.log(output + eps) * target
            return loss.sum() / (target.sum() + eps)

        target_value = batch['value']

        self.optimizer.zero_grad()
        outputs = self.model(batch['obs'])
        predicted_value = outputs.value

        loss_ = critic_loss(predicted_value, target_value)

        loss_ += actor_loss(outputs.move_policy, batch['move_action'])
        loss_ += actor_loss(outputs.sap_policy, batch['sap_action'])

        return loss_


class PPOLearner(Learner):
    """Implements the PPO learning algorithm."""

    def __init__(self, model: torch.nn.Module, model_old: torch.nn.Module, device: torch.device, learning_rate: float):
        super(PPOLearner, self).__init__(model, device, learning_rate)
        self.model_old = model_old.to(device)
        self._update_old_model()

    def _update_old_model(self):
        self.model_old.load_state_dict(self.model.state_dict())

    def import_(self, filepath):
        checkpoint = torch.load(filepath, self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_old.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps = checkpoint['step']
        return {k: v for (k, v) in checkpoint.items() if k not in ['model_state_dict', 'optimizer_state_dict', 'step']}

    def get_loss(self, outputs: NetworkOutput, batch: dict[str, torch.tensor]):
        def actor_loss(policy_old, policy, adv, action):
            a = adv[:, :, None, None, None] * action
            r = policy / (policy_old + 1e-10)
            r_clipped = torch.clip(r, 0.8, 1.2)
            loss = torch.minimum(r * a, r_clipped * a)

            return loss.sum() / (action.sum() + 1e-5)

        def critic_loss(prediction, target):
            return torch.nn.L1Loss()(prediction, target)

        with torch.no_grad():
            old_policies = self.model_old(batch['obs'])

        target_value = batch['value']
        predicted_value = outputs.value

        loss_ = critic_loss(predicted_value, target_value)

        advantage = predicted_value - target_value
        advantage = advantage.detach()

        loss_ += 0.5 * actor_loss(old_policies.move_policy, outputs.move_policy, advantage, batch['move_action'])
        loss_ += 0.5 * actor_loss(old_policies.sap_policy, outputs.sap_policy, advantage, batch['sap_action'])

        with torch.no_grad():
            self._update_old_model()
        return loss_
