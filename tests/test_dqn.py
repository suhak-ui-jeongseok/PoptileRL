import torch

from poptile_rl.environment import NewGame
from poptile_rl.model.dqn import DQN
from poptile_rl.model.dqn import Agent

if __name__ == '__main__':
    game = NewGame(3, 15, 8)
    game.generate_row()

    dqn = DQN(4, 15, 8)
    print(game.to_tensor())
    print(dqn(game.to_tensor().unsqueeze(0)))

    agent = Agent(dqn, 15 * 8, 0.9, 0.05, 200)

    agent.lookup(game)
