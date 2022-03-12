from typing import Dict

class Config:
    url: str = 'http://s0af.panty.run'

    name_xpath: Dict[str, str] = {
        'animation_toggle': '//*[@id="root"]/div/div/div[1]/button',
        'username_input': '//*[@id="root"]/div/div/input',
        'start_button': '//*[@id="root"]/div/div/div[2]/a[1]/button',
        'game_canvas': '//*[@id="root"]/div/div/canvas',
        'ingame_score_text': '//*[@id="root"]/div/div/span',
        'gameover_score_text': '//*[@id="root"]/div/div/h1'
    }
