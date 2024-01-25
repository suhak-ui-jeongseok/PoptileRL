import numpy as np
import random

import torch
import tqdm
import matplotlib.pyplot as plt

from poptile_rl.environment import NewGame, Engine
from poptile_rl.model.dqn import DQN, Agent, History

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_game(engine: Engine):
    n2c = ['0', '1', '2', ' ']
    for r in reversed(engine.board.state):
        print(*map(lambda i: n2c[i], r))

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200, suppress=True)
    plt.ion()
    ROW = 15
    COL = 8
    dqn = DQN(4, ROW, COL)
    agent = Agent(dqn, ROW * COL, 0, 0, 0)

    # load model
    dqn.load_state_dict(torch.load('./poptile_rl/model/weights/model_10.ckpt'))

    # run game with trained model
    game = NewGame(3, ROW, COL)
    game.generate_row()
    while not game.is_gameover:
        state = game.to_tensor()
        print_game(game)
        # print(agent.lookup_survival(game, device).reshape(ROW, COL))
        print(f'{game.score=} {dqn(state.unsqueeze(0).to(device)).item()=}')
        action = agent.select_action_2(game, device, thr=0.2)
        game.pop_tile(action // COL, action % COL)
    else:
        print(game.score)
