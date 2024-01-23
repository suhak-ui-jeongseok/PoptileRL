import torch

from poptile_rl.environment import NewGame
from poptile_rl.model.dqn import History, DQN, Agent


def selfplay(history: History):
    ROW = 15
    COL = 8
    game = NewGame(3, ROW, COL)
    game.generate_row()

    dqn = DQN(4, ROW, COL)
    agent = Agent(dqn, ROW * COL, 0.9, 0.05, 200)

    while not game.is_gameover:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state = game.to_tensor().to(device)
        action = agent.select_action(game, device)
        print(action)
        game.pop_tile(action // COL, action % COL)
        history.push(state, game.score)
    history.new_game()
    return history



if __name__ == '__main__':
    history = History()
    for _ in range(2):
        history = selfplay(history)
    print(history.to_train_data()[0])
