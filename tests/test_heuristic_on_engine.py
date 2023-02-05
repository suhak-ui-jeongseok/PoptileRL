import argparse
import time

from poptile_rl.environment.engine import Engine, NewGame
from poptile_rl.model.heuristic_baseline import search_best


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play", type=int, default=1)
    parser.add_argument("--touch", type=int, default=1000)
    parser.add_argument("--print-board", type=bool, default=False)
    return parser.parse_args()


def print_game(engine: Engine):
    n2c = ["0", "1", "2", " "]
    for r in reversed(engine.board.state):
        print(*map(lambda i: n2c[i], r))


if __name__ == "__main__":
    args = parse_args()
    play: int = args.play
    touch: int = args.touch
    print_board: bool = args.print_board

    for i_play in range(play):
        start_time = time.time()
        game = NewGame(3, 15, 8)
        game.generate_row()
        for i in range(touch):
            if print_board:
                print("--START--")
                print(f"level {i}:")
                print_game(game)
                print("---END---")
            best_action = search_best(game)

            if i >= 10:
                best_action = (0, 0)

            game.pop_tile(*best_action)
            if game.is_gameover:
                print(f"Play {i_play} - score: {game.score}")
                print(f"Avg Time per Iter - {(time.time() - start_time) / i * 1000}")
                break
