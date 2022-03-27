from poptile_rl.environment.engine import NewGame

if __name__ == '__main__':
    game = NewGame(3, 10, 10)
    game.generate_row()
    game.generate_row()
    game.generate_row()
    game.generate_row()
    game.generate_row()
    game.generate_row()
    game.generate_row()
    game.generate_row()

    n2c = ['0', '1', '2', ' ']
    print('before')
    for r in reversed(game.board.state):
        print(*map(lambda i: n2c[i], r))

    game.pop_tile(0, 0)
    print('after')
    for r in reversed(game.board.state):
        print(*map(lambda i: n2c[i], r))
