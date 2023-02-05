import time

from poptile_rl.environment.engine import Engine, NewGame
from poptile_rl.model.heuristic_baseline import search_best


def print_game(engine: Engine):
    n2c = ['0', '1', '2', ' ']
    for r in reversed(engine.board.state):
        print(*map(lambda i: n2c[i], r))


if __name__ == '__main__':
    for trial in range(10):
        start_time = time.time()
        game = NewGame(3, 15, 8)
        game.generate_row()
        for i in range(1000000):
            if True:
                print('--START--')
                print(f'level {i}:')
                print_game(game)
                print('---END---')
            best_action = search_best(game)

            if i >= 10:
                best_action = (0, 0)
            
            game.pop_tile(*best_action)
            if game.is_gameover:
                print(f'Trial {trial} - score: {game.score}')
                print(f'Avg Time per Iter - {(time.time() - start_time) / i * 1000}')
                break
    