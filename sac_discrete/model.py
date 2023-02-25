import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import is_image_space


class MLP(nn.Module):
    '''
    Multi-layer perceptron.

    :param inputs_dim: 1D Input dimension of the input data
    :param outputs_dim: output dimension
    :param n_layer: total number of layer in MLP, minimum is two
    :param n_unit: dimensions of hidden layers
    '''
    def __init__(self, inputs_dim:int, outputs_dim:int, n_layer:int, n_unit=256):
        super().__init__()
        self.inputs_dim     = inputs_dim
        self.outputs_dim    = outputs_dim

        net = [nn.Linear(inputs_dim, n_unit), nn.ReLU()]
        for _ in range(n_layer-2):
            net.append(nn.Linear(n_unit, n_unit))
            net.append(nn.ReLU())
        net.append(nn.Linear(n_unit, outputs_dim))

        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    """
    Modified from `cleanRL`: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari.py
    """
    def __init__(self, inputs_dim: int, outputs_dim:int):
        """
        `inputs_dim` is the channel dimension of the input image.
        In `atari` setting with grayscale and stacked frames of 4, this dimension is set to 4
        `outputs_dim` is the output of the CNN, usually the action space size
        """
        super().__init__()
        # the input images' sizes are required fixed at [84x84]
        self.network = nn.Sequential(
            nn.Conv2d(inputs_dim, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, outputs_dim),
        )

    def forward(self, input):
        return self.network(input)

class Actor(nn.Module):
    '''
    Actor class for state 1d inputs
    '''
    def __init__(self,
            inputs_dim: np.ndarray,
            output_dims: int,
            n_layer: int,
            n_unit: int
    ):
        super().__init__()
        self.inputs_dim     = inputs_dim
        self.output_dims    = np.prod(output_dims) # always int, since the action space is discrete

        self.is_image_space = is_image_space(self.inputs_dim)
        if not self.is_image_space:
            self._actor = MLP(inputs_dim, output_dims, n_layer, n_unit)
        else:
            self._actor = CNN(inputs_dim[-3], output_dims)

    def forward(self, x):
        return self._actor(x)

    def probs(self, x, compute_log_pi=False):
        logits = self.forward(x)
        probs  = F.softmax(logits, dim=1)
        assert len(probs.shape) <= 2
        if not compute_log_pi: return probs, None
        distribution = Categorical(logits=logits)
        entropy = distribution.entropy()
        return probs, entropy

    def entropy(self, x):
        logits = self.forward(x)
        distribution = Categorical(logits=logits)
        return distribution.entropy()

    def sample(self, x, compute_log_pi=False, deterministic=False):
        '''
        Sample action from policy, return sampled actions and log prob of that action
        In inference time, set the sampled actions to be deterministic

        :param x: observation with type `torch.Tensor`
        :param compute_log_pi: return the log prob of action taken
        :param deterministic: return a deterministic or random action
        '''
        logits = self.forward(x)
        distribution = Categorical(logits=logits)

        if deterministic: return torch.argmax(distribution.probs, dim=1), None

        sampled_action  = distribution.sample()

        if not compute_log_pi: return sampled_action, None

        entropy = distribution.entropy()
        return sampled_action, entropy

class DoubleQNet(nn.Module):
    def __init__(self, state_dim, action_dim, n_layer, n_unit):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_image_space = is_image_space(self.state_dim)
        if not self.is_image_space:
            self.q1 = MLP(state_dim, action_dim, n_layer, n_unit)
            self.q2 = MLP(state_dim, action_dim, n_layer, n_unit)
        else:
            self._actor = CNN(inputs_dim[-3], output_dims)
            self.q1 = CNN(state_dim, action_dim, n_layer, n_unit)
            self.q2 = CNN(state_dim, action_dim, n_layer, n_unit)

    def forward(self, x):
        return self.q1(x), self.q2(x)

class Critic(nn.Module):
    def __init__(self, obs_shape, action_shape, n_layer, n_unit):
        super().__init__()

        self._online_q = DoubleQNet(obs_shape, action_shape, n_layer, n_unit)
        self._target_q = DoubleQNet(obs_shape, action_shape, n_layer, n_unit)

        self._target_q.load_state_dict(self._online_q.state_dict())

    def target_q(self, x): return self._target_q(x)

    def online_q(self, x): return self._online_q(x)

    def polyak_update(self, tau):
        '''Exponential evaraging of the online q network'''
        for target, online in zip(self._target_q.parameters(), self._online_q.parameters()):
            target.data.copy_(target.data * (1-tau) + tau * online.data)
