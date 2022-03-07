import base64
import io

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome import service as Service
from selenium.webdriver.common import action_chains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

class Manager:
    def __init__(self, driver_path, username):
        self.username = username

        # TODO: delete after finishing modulization
        driver_path = 'C:/Users/kunwo/Documents/chromedriver/chromedriver.exe'

        service = Service.Service(executable_path=driver_path)
        self.driver = webdriver.Chrome(service=service)
        self.driver.get(url='http://s0af.panty.run/')

        self.start_game()

        self.canvas_element = self.get_canvas_element()


    def start_game(self):
        '''
        1. toggle animation by click
        2. set username by send_keys
        3. press start button by click
        '''
        try:
            xpath_list = [
                '//*[@id="root"]/div/div/div[1]/button',
                '//*[@id="root"]/div/div/input',
                '//*[@id="root"]/div/div/div[2]/a[1]/button',
            ]
            action_list = [
                lambda element: element.click(),
                lambda element: element.send_key('Test'),
                lambda element: element.click(),
            ]
            for xpath, action in zip(xpath_list, action_list):
                selector = (By.XPATH, xpath)
                element = WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located(selector)
                )
                action(element)

        except Exception as e:
            print(e)


    def quit(self):
        self.driver.close()


    def get_canvas_element(self):
        canvas_element = None

        try:
            canvas_selector = (
                By.XPATH,
                '//*[@id="root"]/div/div/canvas'
            )
            canvas_element = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located(canvas_selector)
            )
        except Exception as e:
            print(e)

        return canvas_element


    def get_tile_matrix(self):
        try:
            b64_canvas = self.canvas_element.screenshot_as_base64

            img = Image.open(io.BytesIO(base64.b64decode(b64_canvas)))
            for y in range(15):
                for x in range(8):
                    pixel_pos = (x * 30 + 15), ((14 - y) * 30 + 15)
                    print(img.getpixel(pixel_pos))
        except Exception as e:
            print(e)


    def pop_tile(self, x, y):
        pointer_offset = (x * 31 + 15), ((14 - y) * 31 + 15)

        try:
            clickTile = action_chains.ActionChains(self.driver)\
                .move_to_element_with_offset(self.canvas_element, *pointer_offset)\
                .click_and_hold()\
                .pause(0.05)\
                .release()

            clickTile.perform()

            print(pointer_offset, 'clicked')
        except Exception as e:
            print(e)
