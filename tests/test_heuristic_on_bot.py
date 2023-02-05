from time import sleep
from typing import List

from poptile_rl.bot.bot import Bot
from poptile_rl.environment.engine import Engine
from poptile_rl.environment.board import Board
from poptile_rl.model.heuristic_baseline import search_best
from poptile_rl.config import Config


def rgb_to_board(rgb_matrix: List[List[int]]) -> List[List[int]]:
    rgb2id = {
        (255, 255, 255): -1,
        (255, 171, 0): 0,
        (0, 255, 171): 1,
        (0, 171, 255): 2
    }
    return [[rgb2id.get(e, None) for e in line] for line in rgb_matrix]


def print_game(data):
    n2c = ['0', '1', '2', ' ']
    for r in reversed(data):
        print(*map(lambda i: n2c[i], r))

def run():
    bot_agent = Bot(Config.driver_path, Config.url, 'Magister', Config.name_xpath)

    last_id_matrix = None
    for i in range(100000):
        print(f'Iteration {i}')
        while True:
            rgb_matrix = bot_agent.get_tile_matrix()
            id_matrix = rgb_to_board(rgb_matrix)
            if last_id_matrix != id_matrix:
                break
        print_game(id_matrix)

        engine = Engine(-1, Board(3, 15, 8, id_matrix))
        best_action = search_best(engine)

        if i >= 4000:
            best_action = (0, 0)

        bot_agent.poptile(best_action)

        sleep(0.3)

        if bot_agent.is_gameover():
            break

    else:
        input()

    bot_agent.quit()



if __name__ == '__main__':
    for i in range(10):
        run()
