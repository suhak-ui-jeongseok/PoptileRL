from collections import deque, namedtuple
from random import sample, random, randrange, choice
from math import exp
from typing import List, Tuple

import numpy as np
import torch
from torch import nn, tensor, Tensor, LongTensor
from torch.nn import Softmax

from poptile_rl.environment import Board, Engine
from poptile_rl.model.heuristic_baseline import search_best, select_random

class History:
    def __init__(self) -> None:
        self.games: list[list] = []
        self.now_game = None
        self.transitions = namedtuple('Game', ('state', 'score'))
    
    def push(self, state, score):
        if self.now_game is None:
            self.now_game = []
        self.now_game.append(self.transitions(state, score))
    

    def new_game(self):
        self.games.append(self.now_game)
        if len(self.games) > 100:
            self.games.pop(0)
        self.now_game = None


    def to_train_data(self, *, only_survive=False):
        a_type = []
        b_type = []
        for game in self.games:
            for idx, t in enumerate(game):
                # look 15 steps of score ahead. 
                # input: state, output: score after 15 steps.
                # if game is over after 15 steps, output: 0
                if idx + 7 >= len(game):
                    b_type.append((t.state, 0))
                else:
                    if only_survive:
                        a_type.append((t.state, 1))
                    else:
                        a_type.append((t.state, game[idx + 6].score - t.score))
        return a_type, b_type
    
class DatasetBuilder:
    def __init__(self, history: History):
        self.history = history
    
    def survival_rate_lookup(self, step: int):
        a_type = []
        b_type = []
        for game in self.history.games:
            for idx, t in enumerate(game):
                # look n-steps of state(gameover or on play) ahead.
                # 0: game over, 1: survive
                if idx + step >= len(game):
                    b_type.append((t.state, 0))
                else:
                    a_type.append((t.state, 1))
        return a_type, b_type
    
    def survive_n_step(self, max_step: int):
        a_type = [] # not max_step
        b_type = [] # max_step
        for game in self.history.games:
            for idx, t in enumerate(game):
                # look n-steps of state(gameover or on play) ahead.
                # 0: game over after 1 step
                # n: would survive n steps
                if idx + max_step >= len(game):
                    b_type.append((t.state, 0))
                else:
                    a_type.append((t.state, max_step))

        return a_type, b_type

class ResBlock(nn.Module):
    def __init__(self, n_channel: int):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
            nn.Conv2d(n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(n_channel),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.relu(x + self.conv(x))


class DQN(nn.Module):
    '''
    DQN. input: [batch x color x r x c], output: [1](q-value)
    '''
    def __init__(self, 
                 n_channel: int,
                 row: int,
                 col: int):
        super(DQN, self).__init__()

        self.n_channel: int = n_channel
        self.row: int = row
        self.col: int = col
        
        N_RESBLOCK_CHANNEL = 512

        self.starting = nn.Sequential(
            nn.Conv2d(self.n_channel, N_RESBLOCK_CHANNEL, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.BatchNorm2d(N_RESBLOCK_CHANNEL),
            nn.ReLU(),
        )

        self.conv = nn.Sequential(
            *[ResBlock(N_RESBLOCK_CHANNEL) for _ in range(5)]
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(N_RESBLOCK_CHANNEL, 32, kernel_size=(1, 1), stride=(1, 1), padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # conv_w = self.conv2d_output_shape(8, kernel_size=3, stride=1)
        # conv_h = self.conv2d_output_shape(15, kernel_size=3, stride=1)
        # linear_input_shape = conv_w * conv_h * 32
        linear_input_shape = 8 * 15 * 4

        self.head = nn.Sequential(
            nn.Linear(linear_input_shape, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        # x = self.starting(x)
        # x = self.conv(x)
        # x = self.bottleneck(x)
        x = x.flatten(start_dim=1)
        x = self.head(x)
        return x
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

    def lookup(self, engine: Engine, device=torch.device('cpu')):
        # try all actions possible, and return expectation vector
        expectation = np.zeros(self.n_action)
        for i in range(self.n_action):
            if engine.board[i // 8, i % 8] == -1:
                expectation[i] = -1
                continue
            new_engine = engine.copy()
            new_engine.pop_tile(i // 8, i % 8)
            if new_engine.is_gameover:
                expectation[i] = 0
            else:
                input_tensor = new_engine.board.to_tensor().squeeze(0).unsqueeze(0).to(device)
                lookup_score = self.dqn(input_tensor).item()
                expectation[i] = np.exp(lookup_score * 10) + new_engine.score - engine.score

        return expectation
    
    def lookup2(self, engine: Engine, device=torch.device('cpu')):
        # try all actions possible, and return expectation vector
        expectation = np.zeros(self.n_action)
        q_values = np.zeros(self.n_action)
        for i in range(self.n_action):
            if engine.board[i // 8, i % 8] == -1:
                expectation[i] = -1
                continue
            new_engine = engine.copy()
            new_engine.pop_tile(i // 8, i % 8)
            if new_engine.is_gameover:
                expectation[i] = 0
            else:
                input_tensor = new_engine.board.to_tensor().squeeze(0).unsqueeze(0).to(device)
                lookup_score = self.dqn(input_tensor).item()
                expectation[i] = np.exp(lookup_score) + new_engine.score - engine.score
                q_values[i] = lookup_score

        return expectation, q_values
    
    
    def lookup_survival(self, engine: Engine, device=torch.device('cpu')):
        # try all actions possible, and return expectation vector
        expectation = np.ones(self.n_action)
        input_tensor = tensor(np.zeros((self.n_action, 4, 15, 8)), dtype=torch.float).to(device)
        for i in range(self.n_action):
            if engine.board[i // 8, i % 8] == -1:
                expectation[i] = -1
                continue
            new_engine = engine.copy()
            new_engine.pop_tile(i // 8, i % 8)
            if new_engine.is_gameover:
                expectation[i] = 0
                continue
            input_tensor[i] = new_engine.board.to_tensor()

        lookup_score = torch.sigmoid(self.dqn(input_tensor)).squeeze(1)
        # print(lookup_score)
        for i in range(self.n_action):
            if expectation[i] == -1 or expectation[i] == 0:
                continue
            expectation[i] = lookup_score[i].item()
        return expectation
    
    def eps(self):
        s, e, d, sd = self.eps_start, self.eps_end, self.eps_decay, self.steps_done

        return min(max((s - e) * (d - sd) / d + e, 0.0), 1.0)

    
    def select_action(self, engine: Engine, device, *, select_best=False, survival=False) -> int:
        eps = self.eps()

        # if random() < 0.001:
        #     print(lookup)

        if random() < eps and not select_best:
            r, c = select_best(engine)
            return r * 8 + c
            # lookup = self.lookup(engine, device)
            # return choice(np.where(lookup != -1)[0])
        else:
            if survival:
                lookup = self.lookup_survival(engine, device)
            else:
                lookup = self.lookup(engine, device)
        
            return lookup.argmax()
    
    def select_action_2(self, engine: Engine, device, thr = -0.9) -> int:
        lookup = tensor(self.lookup_survival(engine, device))
        lookup_2nd = np.zeros_like(lookup)
        
        # for all positive lookups, look up again
        # for i in np.where(lookup > thr)[0]:
        #     new_engine = engine.copy()
        #     new_engine.pop_tile(i // 8, i % 8)
        #     lookup_2nd[i] = (self.lookup_survival(new_engine, device)).max()
        
        print(lookup.numpy().reshape(15, 8)[::-1, ::])
        # print(lookup_2nd.reshape(15, 8)[::-1, ::])
        
        return lookup.argmax()