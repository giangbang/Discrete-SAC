import numpy as np
import torch
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('states', 'actions', 'rewards', 'next_states', 'dones'))

# https://github.com/denisyarats/pytorch_sac_ae/blob/master/utils.py
class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device='auto'):
        self.capacity = capacity
        self.batch_size = batch_size
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            device = self.device
        else:
            self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 # if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, 1), dtype=np.int64)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, terminated, truncated, info=None):
        '''Add a new transition to replay buffer'''
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], terminated)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self):
        '''Sample batch of Transitions with batch_size elements.
        Return a named tuple with 'states', 'actions', 'rewards', 'next_states' and 'dones'. '''
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        dones = torch.as_tensor(self.dones[idxs], device=self.device)

        return Transition(obses, actions, rewards, next_obses, dones)
