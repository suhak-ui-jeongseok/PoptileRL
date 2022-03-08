from bot import bot_manager


if __name__ == '__main__':
    my_bot = bot_manager.Manager(
        driver_path='C:/Users/kunwo/Documents/chromedriver/chromedriver.exe',
        username='bot_test'
    )
    
    while True:
        rgb_matrix = my_bot.get_tile_matrix()
        for l in my_bot.get_tile_matrix():
            print(l)

        prompt = input()
        if prompt == '':
            x, y = 0, 0
        else:
            x, y = map(int, prompt.split(' '))
        
        is_gameover, score = my_bot.poptile((x, y))
        print(score)        
        if is_gameover:
            break
    
    my_bot.quit()