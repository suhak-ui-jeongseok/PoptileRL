import numpy as np
import random

import torch
import tqdm
import matplotlib.pyplot as plt

from poptile_rl.environment import NewGame
from poptile_rl.model.dqn import DQN, Agent, History, DatasetBuilder


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


episode_scores = []
def plot_scores():
    plt.figure(1)
    plt.clf()
    plt.scatter([i // N_GAMEPLAY for i in range(len(episode_scores))], episode_scores, s=50)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.show()
    plt.pause(0.1)


def data_inspection(a_type, b_type, mode='binary'):
    print(len(a_type), len(b_type))
    plt.figure(2)
    plt.clf()
    plt.title('Data distribution')
    if mode == 'binary':
        a = np.array([a[1] for a in a_type])
        b = np.array([b[1] for b in b_type])
        plt.hist(a, bins=20, alpha=0.5, label='Survive')
        plt.hist(b, bins=20, alpha=0.5, label='Gameover')
        plt.legend()
    elif mode == 'log':
        a = np.array([a[1] for a in a_type])
        plt.hist(np.log(a), bins=20)
    
    plt.show()
    plt.pause(0.5)


def model_inspection(dataset: tuple, dqn: DQN):
    # sample some data, and see distribution of actual model q-values
    a_type, b_type = dataset
    if len(a_type) > 500:
        a_type = random.sample(a_type, 500)
    if len(b_type) > 500:
        b_type = random.sample(b_type, 500)

    # state_batch, score_batch = zip(*batch)
    # state_batch = torch.stack(state_batch).to(device)
    # score_batch = torch.tensor(score_batch, dtype=torch.float).unsqueeze(1).to(device)
    
    a_state_batch = torch.stack([a[0] for a in a_type]).to(device)
    b_state_batch = torch.stack([b[0] for b in b_type]).to(device)

    with torch.no_grad():
        a_q = dqn(a_state_batch)
        b_q = dqn(b_state_batch)

    BINARY_MODE = True
    if BINARY_MODE:
        a_q = torch.sigmoid(a_q)
        b_q = torch.sigmoid(b_q)
    
    a_q = a_q.squeeze(1).numpy()
    b_q = b_q.squeeze(1).numpy()
    
    plt.figure(3)
    plt.clf()
    plt.title('Model inspection')
    plt.hist(a_q, bins=25, alpha=0.5, label='Survive')
    plt.hist(b_q, bins=25, alpha=0.5, label='Gameover')
    plt.legend()
    plt.show()
    plt.pause(0.5)


def selfplay(history: History, agent: Agent, survival=False, iterator=None):
    ROW = 15
    COL = 8
    game = NewGame(3, ROW, COL)
    game.generate_row()

    while not game.is_gameover:
        state = game.to_tensor()
        action = agent.select_action(game, device, survival=survival)
        game.pop_tile(action // COL, action % COL)
        history.push(state, game.score)
        if iterator is not None:
            iterator.set_postfix_str(f'{game.score=}')
    else:
        episode_scores.append(game.score)
        history.new_game()


def optimize_model(
        history: History, 
        dqn: DQN,
        optimizer,
        loss,
        batch_size: int):
    
    a_type, b_type = history.to_train_data()

    batch = random.sample(a_type, batch_size // 2) + random.sample(b_type, batch_size // 2)
    state_batch, score_batch = zip(*batch)
    state_batch = torch.stack(state_batch).to(device)
    score_batch = torch.tensor(score_batch, dtype=torch.float).unsqueeze(1).to(device)
    score_batch = torch.log(score_batch + 1e-4) / 10
    # print(state_batch.shape, score_batch.shape)

    dqn.train()
    optimizer.zero_grad()
    output = dqn(state_batch)
    loss_value = loss(output, score_batch)
    loss_value.backward()
    optimizer.step()
    return loss_value.item()


def optimize_model_survive(
        data, 
        dqn: DQN, 
        optimizer,
        loss,
        batch_size: int):
    
    a_type, b_type = data

    batch = random.sample(a_type, batch_size // 2) + random.sample(b_type, batch_size // 2)
    state_batch, score_batch = zip(*batch)
    state_batch = torch.stack(state_batch).to(device)
    score_batch = torch.tensor(score_batch, dtype=torch.float).unsqueeze(1).to(device)
    # print(state_batch.shape, score_batch.shape)

    dqn.train()
    optimizer.zero_grad()
    output = dqn(state_batch)
    loss_value = loss(output, score_batch)
    loss_value.backward()
    optimizer.step()
    return loss_value.item()

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=200)
    plt.ion()
    ROW = 15
    COL = 8
    N_EPISODE = 5
    N_LEARNSTEP = 400
    N_GAMEPLAY = 100
    dqn = DQN(4, ROW, COL)
    agent = Agent(dqn, ROW * COL, 1.0, 0.0, N_EPISODE)
    history = History()
    dataset_builder = DatasetBuilder(history)

    optimizer = torch.optim.Adam(dqn.parameters(), lr=0.001)
    loss = torch.nn.MSELoss()
    loss_survive = torch.nn.BCEWithLogitsLoss()
    # loss_survive = torch.nn.MSELoss()

    # selfplay and optimize
    for i_episode in range(N_EPISODE + 5):
        print(f'episode {i_episode + 1}, eps: {agent.eps():.4f}')
        iterator = tqdm.trange(N_GAMEPLAY, desc=f'Episode {i_episode + 1}, selfplay')
        for _ in iterator:
            selfplay(history, agent, iterator=iterator)
        

        # Instpection START
        plot_scores()
        a_type, b_type = dataset_builder.survival_rate_lookup(10)
        data_inspection(a_type, b_type)
        model_inspection((a_type, b_type), dqn)
        # Instpection END

        iterator = tqdm.trange(N_LEARNSTEP, desc=f'Episode {i_episode + 1}, train')
        for _ in iterator:
            loss_value = optimize_model_survive((a_type, b_type), dqn, optimizer, loss_survive, 32)
            iterator.set_postfix_str(f'loss: {loss_value:.4f}')

        # save model 
        ckpt_file_name = f'./poptile_rl/model/weights/model_{i_episode + 1}.ckpt'
        torch.save(dqn.state_dict(), ckpt_file_name)

        agent.steps_done += 1
    else:
        plt.ioff()
        plot_scores()
    
    # run game with trained model
    game = NewGame(3, ROW, COL)
    game.generate_row()
    while not game.is_gameover:
        state = game.to_tensor()
        print(agent.lookup_survival(game, device).reshape(ROW, COL))
        print(f'{game.score=} {dqn(state.unsqueeze(0).to(device)).item()=}')
        action = agent.select_action_2(game, device, thr=-1)
        game.pop_tile(action // COL, action % COL)
    else:
        print(game.score)   

