from collections import namedtuple, deque
from random import sample, random, randrange, choice
from math import exp
from typing import List, Tuple

import torch
from torch import nn, tensor, Tensor, LongTensor
from torch.nn import Softmax


class ReplayMemory:
    def __init__(self, batch_size: int = 1, max_len: int = 10000):
        self.batch_size: int = batch_size
        self.memory: deque = deque([], maxlen=max_len)
        self.transition: namedtuple = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    def __len__(self) -> int:
        return len(self.memory)

    def push(self, *args):
        self.memory.append(self.transition(*args))

    def sample(self) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        transition = sample(self.memory, self.batch_size)
        batch: transition = self.transition(*zip(*transition))

        state: Tensor = Tensor(batch.state).cuda()
        action: Tensor = Tensor(batch.action).cuda()
        next_state: Tensor = Tensor(batch.next_state).cuda()
        reward: Tensor = Tensor(batch.reward).cuda()
        return state, action, next_state, reward


class DQN(nn.Module):
    def __init__(self, row: int,
                 col: int,
                 output_dim: int):
        super(DQN, self).__init__()

        self.row: int = row
        self.col: int = col
        self.output_dim: int = output_dim

        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 5), stride=(1, 1), padding=1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(5, 5), stride=(1, 1), padding=1),
            nn.BatchNorm2d(32)
        )

        conv_w = self.conv2d_output_shape(row)
        conv_h = self.conv2d_output_shape(col)
        linear_input_shape = conv_w * conv_h * 32

        self.linear = nn.Sequential(
            nn.Linear(col, 128),
            nn.Linear(128, 128),
            nn.Linear(128, 128),
            nn.Linear(128, col)
        )
        self.head = nn.Linear(linear_input_shape, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x = x.cuda()
        # x = self.relu(self.bn1(self.conv1(x)))
        # x = self.relu(self.bn2(self.conv2(x)))
        # x = self.relu(self.bn3(self.conv3(x)))
        x = x.cuda()
        x = self.linear(x)
        return self.relu(x.reshape(-1, self.row, self.col))
        # return self.relu(self.head(x.flatten()))

    @staticmethod
    def conv2d_output_shape(size, kernel_size=5, stride=2):
        return (size - (kernel_size - 1) - 1) // stride + 2


class Agent(nn.Module):
    """
    Get State -> DQN -> Get Action -> Env -> Reward + Next State
    ReplayMemory (Current State, Action, Next State, Reward)

    State: [row, col]
    Action: [row x col]
    Next State: [row, col]
    Reward: int

    """
    def __init__(self, dqn: DQN,
                 n_action: int,
                 eps_start: float,
                 eps_end: float,
                 eps_decay: float):
        super(Agent, self).__init__()
        self.dqn: DQN = dqn
        self.n_action: int = n_action

        self.eps_start: float = eps_start
        self.eps_end: float = eps_end
        self.eps_decay: float = eps_decay
        self.steps_done: int = 0

    def forward(self, state: Tensor) -> Tensor:
        action: Tensor = self.action(state)
        return action

    def action(self, state: Tensor) -> Tensor:
        sample_prob: float = random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample_prob > eps_threshold:
            action_output = self.dqn(state)
            masked_action_output = action_output.masked_fill(state == -1, -1e6).cuda()
            masked_action_output: Tensor = masked_action_output.reshape(-1, self.n_action).argmax(1).cuda()
            # print(masked_action_output)
            return masked_action_output
        else:
            masked_action_output: List[int] = (state.reshape(-1, self.n_action) >= 0).nonzero(as_tuple=True)[0].tolist()
            action: int = choice(masked_action_output)
            action: Tensor = LongTensor([action]).cuda()
            # print('rand', action)
            return action
