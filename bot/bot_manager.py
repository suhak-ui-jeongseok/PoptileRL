import base64
import io

from typing import List, Tuple
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome import service as Service
from selenium.webdriver.common import action_chains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


class Manager:
    def __init__(self, driver_path:str, username:str):
        self.username:str = username

        options = webdriver.ChromeOptions()
        options.add_argument('--incognito')
        options.add_argument('--no-sandbox')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        service = Service.Service(executable_path=driver_path)

        self.__driver = webdriver.Chrome(service=service, chrome_options=options)

        self.start_game()

        self.score = 0

        self.__canvas_element = self.get_canvas_element()


    def start_game(self):
        '''
        1. toggle animation by click
        2. set username by send_keys
        3. press start button by click
        '''
        self.__driver.get(url='http://s0af.panty.run/')

        try:
            xpath_list = [
                '//*[@id="root"]/div/div/div[1]/button',
                '//*[@id="root"]/div/div/input',
                '//*[@id="root"]/div/div/div[2]/a[1]/button',
            ]
            action_list = [
                lambda element: element.click(),
                lambda element: element.send_keys(self.username),
                lambda element: element.click(),
            ]
            for xpath, action in zip(xpath_list, action_list):
                selector = (By.XPATH, xpath)
                element = WebDriverWait(self.__driver, 10).until(
                    EC.presence_of_element_located(selector)
                )
                action(element)

        except Exception as e:
            print(e)


    def quit(self):
        self.__driver.close()
        self.__driver.quit()


    def get_canvas_element(self):
        canvas_element = None

        try:
            canvas_selector = (
                By.XPATH,
                '//*[@id="root"]/div/div/canvas'
            )
            canvas_element = WebDriverWait(self.__driver, 10, 0.1).until(
                EC.presence_of_element_located(canvas_selector)
            )
        except Exception as e:
            print(e)

        return canvas_element

    def is_gameover(self):
        print(self.__driver.current_url)
        return self.__driver.current_url == 'http://s0af.panty.run/single/result'


    def get_tile_matrix(self):
        try:
            b64_canvas = self.__canvas_element.screenshot_as_base64
        except Exception as e:
            print(e)

        matrix = []
        img = Image.open(io.BytesIO(base64.b64decode(b64_canvas)))
        matrix:List[List[Tuple]]
        for y in range(15):
            row = []

            for x in range(8):
                pixel_pos = (x * 30 + 15), ((14 - y) * 30 + 15)
                row.append(img.getpixel(pixel_pos)[:3])

            matrix.append(row)

        return matrix


    def poptile(self, pos:Tuple[int, int]):
        x, y = pos
        pointer_offset = (x * 30 + 15), ((14 - y) * 30 + 15)

        try:
            clickTile = action_chains.ActionChains(self.__driver)\
                .move_to_element_with_offset(self.__canvas_element, *pointer_offset)\
                .click()
            clickTile.perform()

            print(pointer_offset, 'clicked')
        except Exception as e:
            print(e)

        return self.get_score()

    def get_score(self):
        a = WebDriverWait(self.__driver, 10).until(
            self.changed_score
        )
        print(a)
        is_gameover, score = a
        return is_gameover, score

    
    def changed_score(self, driver: webdriver.Chrome):
        ingame_locator = (By.XPATH, '//*[@id="root"]/div/div/span')
        gameover_locator = (By.XPATH, '//*[@id="root"]/div/div/h1')
        ingame_element = driver.find_elements(*ingame_locator)
        gameover_element = driver.find_elements(*gameover_locator)

        if ingame_element:
            current_score = int(ingame_element[0].text.split(' ')[2])
            if self.score != current_score:
                self.score = current_score
                return False, self.score
            else:
                return False
        elif gameover_element:
            current_score = int(gameover_element[0].text)
            self.score = current_score
            return True, self.score
        else:
            return False



