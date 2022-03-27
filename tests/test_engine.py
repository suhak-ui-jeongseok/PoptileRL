from poptile_rl.environment.engine import NewGame


def print_game(data):
    n2c = ['0', '1', '2', ' ']
    for r in reversed(data):
        print(*map(lambda i: n2c[i], r))


if __name__ == '__main__':
    game = NewGame(3, 15, 8)
    game.generate_row()
    for i in range(1000):
        print(f'State {i}:')
        print_game(game.board.state_as_array())
        game.pop_tile(0, i % 8)
        if game.is_gameover():
            break
    
    print(f'score: {game.score}')