from typing import List

from poptile_rl.bot.bot import Bot
from poptile_rl.config import Config


def rgb_to_board(rgb_matrix: List[List[int]]) -> List[List[int]]:
    rgb2id = { 
        (255, 255, 255): -1,
        (255, 171, 0): 0,
        (0, 255, 171): 1,
        (0, 171, 255): 2
    }
    return [[rgb2id[e] for e in line] for line in rgb_matrix]


def print_game(data):
    n2c = ['0', '1', '2', ' ']
    for r in reversed(data):
        print(*map(lambda i: n2c[i], r))


if __name__ == '__main__':
    bot_agent = Bot(driver_path=Config.driver_path, username='bot_test', url=Config.url, name_xpath=Config.name_xpath)

    for i in range(14):
        rgb_matrix = bot_agent.get_tile_matrix()
        id_matrix = rgb_to_board(rgb_matrix)
        print_game(id_matrix)

        bot_agent.poptile((0, 0))
    else:
        input()

    bot_agent.quit()
