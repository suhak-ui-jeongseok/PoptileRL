from poptile_rl.environment.engine import NewGame
from poptile_rl.model.heuristic_baseline import search_best


def print_game(engine):
    n2c = ['0', '1', '2', ' ']
    for r in reversed(game.board.state):
        print(*map(lambda i: n2c[i], r))


if __name__ == '__main__':
    game = NewGame(3, 15, 8)
    game.generate_row()
    for i in range(1000):
        print('--START--')
        print(f'level {i}:')
        print_game(game)
        print('---END---')
        pos_r, pos_c = search_best(game)
        game.pop_tile(pos_r, pos_c)
        if game.is_gameover():
            print(f'score: {game.score}')
            break
    