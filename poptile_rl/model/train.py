from random import random, randrange
from typing import List
from math import exp

from numpy import array
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor, zeros
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

from environment import Board, Engine
from DQN import DQN, ReplayMemory, Agent
from config import h_params

steps_done = 0

writer = SummaryWriter()
viz = Visdom()


def plot_durations(episode_durations: List[float]):
    plt.figure()
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(durations_t.numpy())

    # 100개의 에피소드 평균을 가져 와서 도표 그리기

    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤


def train():
    episodes: int = h_params['EPISODE']
    lr: float = h_params['LR']
    batch_size: int = h_params['BATCH_SIZE']
    discount_factor: float = h_params['DISCOUNT_FACTOR']

    n_row: int = 15
    n_col: int = 8
    n_color: int = 3

    dqn = DQN(row=n_row,
              col=n_col,
              output_dim=n_row*n_col).cuda()

    agent = Agent(dqn=dqn,
                  n_action=n_row*n_col,
                  eps_start=h_params['EPS_START'],
                  eps_end=h_params['EPS_END'],
                  eps_decay=h_params['EPS_DECAY']).cuda()

    optimizer = Adam(params=(dqn.parameters()), lr=lr)
    criterion = nn.SmoothL1Loss().cuda()

    score_history: List[float] = []

    for episode in tqdm(range(1, episodes+1)):
        replay_memory: ReplayMemory = ReplayMemory(batch_size=batch_size)

        board: Board = Board(row=n_row, column=n_col, n_color=n_color)
        env: Engine = Engine(0, board)
        board_state: array = env.board.state
        while env.is_gameover() is False:
            cur_state = Tensor(board_state).cuda()
            action: int = agent(cur_state).item()
            action_row: int = action // n_col
            action_col: int = action % n_col
            reward: int = env.pop_tile(action_row, action_col)
            next_state: array = board.state

            replay_memory.push(board_state, action, next_state, reward)
            board_state = next_state
        score_history.append(env.score)

        viz.bar(X=episode, Y=env.score)
        writer.add_scalar('Loss/Train', env.score, episode)
        # plot_durations(score_history)
        # plt.ioff()
        # plt.show()

        # learn
        if len(replay_memory) < batch_size:
            continue

        q_value: Tensor = zeros(batch_size, requires_grad=True).cuda()
        target: Tensor = zeros(batch_size, requires_grad=True).cuda()

        state, action, next_state, reward = replay_memory.sample()
        actions: List[int] = agent(state).detach().tolist()
        state: List[List[int]] = state.detach().tolist()

        for idx, (s, a, n_s, r) in enumerate(zip(state, actions, next_state, reward)):
            learn_board: Board = Board(row=n_row, column=n_col, n_color=n_color, state=s)
            action_row: int = a // n_col
            action_col: int = a % n_col
            score: int = env(learn_board, r, action_row, action_col)
            q_value[idx] = score

        next_actions: List[int] = agent(next_state).tolist()
        next_state: List[List[int]] = next_state.detach().tolist()

        for next_idx, (s, a, n_s, r) in enumerate(zip(state, next_actions, next_state, reward)):
            next_state: Board = Board(row=n_row, column=n_col, n_color=n_color, state=n_s)
            next_action_row: int = a // n_col
            next_action_col: int = a % n_col
            next_q_value = env(next_state, r, next_action_row, next_action_col)
            next_q_value = next_q_value * discount_factor + r
            target[next_idx] = next_q_value

        loss = criterion(q_value, target)

        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.close()


def main():
    print('torch.cuda.is_available()', torch.cuda.is_available())
    train()


main()
