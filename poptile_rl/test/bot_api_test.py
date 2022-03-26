from poptile_rl.bot.bot import Bot
from poptile_rl.config import Config

if __name__ == '__main__':
    bot_agent = Bot(driver_path=Config.driver_path, username='bot_test', url=Config.url, name_xpath=Config.name_xpath)

    for i in range(14):
        rgb_matrix = bot_agent.get_tile_matrix()
        for l in bot_agent.get_tile_matrix():
            print(l)

        bot_agent.poptile((0, 0))
    else:
        input()

    bot_agent.quit()