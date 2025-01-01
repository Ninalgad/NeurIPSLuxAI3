import abc
import torch
from replay import ReplayBuffer


class Learner(metaclass=abc.ABCMeta):
    """An learner to update the network weights based."""

    @abc.abstractmethod
    def learn(self, replay_buffer: ReplayBuffer):
        """Single training step of the learner."""

    @abc.abstractmethod
    def export(self, filepath: str, meta: dict):
        """Exports the learner states."""


def actor_loss(policy, advantage, action):
    a = advantage[:, :, None, None, None] * action
    log_policy = torch.log(policy + 1e-5)
    return - (log_policy * a).mean()


def critic_loss(prediction, target):
    return torch.nn.L1Loss()(prediction, target)


class A2CLearner(Learner):
    """Implements the Advantage Actor Critic (A2C) learning algorithm."""

    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 learning_rate: float):
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.model = model
        self.device = device
        self.model.to(device)
        self.steps = 0

    def learn(self, replay_buffer):
        """Applies a single training step."""
        self.model.train()
        batch = replay_buffer.sample_batch()
        batch = {k: torch.tensor(v).float().to(self.device) for (k, v) in batch.items()}

        target_value = batch['value'].unsqueeze(1)

        self.optimizer.zero_grad()
        outputs = self.model(batch['obs'])
        predicted_value = outputs.value

        assert predicted_value.shape == target_value.shape, (predicted_value.shape, target_value.shape)

        loss_ = critic_loss(predicted_value, target_value)

        advantage = predicted_value - target_value
        advantage = advantage.detach()

        loss_ += 0.5 * actor_loss(outputs.move_policy, advantage, batch['move_action'])
        loss_ += 0.5 * actor_loss(outputs.sap_policy, advantage, batch['sap_action'])
        loss_.backward()

        self.optimizer.step()
        self.steps += 1

        return loss_.detach().cpu().numpy().item()

    def export_model_weights(self, filepath):
        torch.save(self.model.state_dict(), filepath)

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
