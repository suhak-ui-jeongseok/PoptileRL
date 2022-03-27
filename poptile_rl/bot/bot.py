from io import BytesIO
from typing import Dict, List, Tuple

from PIL import Image
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.remote.webelement import WebElement

'''
WebDriverCore driver에 직접 접근
Bot WebDriverCore에 명령 전달, Bot은 동작의 의미를 담고 있어야 함
'''


class WebDriverCore:
    def __init__(self, driver_path: str):
        self.driver = self.init_driver(driver_path)

        self.elements: Dict[str, WebElement] = {}

    @staticmethod
    def init_driver(driver_path: str) -> Chrome:
        options = ChromeOptions()
        options.add_argument('--incognito')
        options.add_argument('--no-sandbox')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])

        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")

        service = Service(executable_path=driver_path)

        return Chrome(service=service, chrome_options=options)

    def quit(self):
        self.driver.close()
        self.driver.quit()

    def get(self, u: str):
        self.driver.get(url=u)

    def wait(self):
        self.driver.implicitly_wait(10)

    def store_elements(self, name: str, xpath: str):
        self.elements[name] = self.driver.find_element(By.XPATH, value=xpath)

    def send_text(self, name: str, text: str):
        self.elements[name].send_keys(text)

    def click(self, name: str):
        self.elements[name].click()

    def click_with_offset(self, name: str, offset: Tuple[int, int]):
        ActionChains(self.driver).move_to_element_with_offset(self.elements[name], *offset).click().perform()

    def get_text_from(self, name: str) -> str:
        return self.elements[name].text

    def get_png_from(self, name: str) -> str:
        return self.elements[name].screenshot_as_png

    def get_url(self) -> str:
        return self.driver.current_url


class Bot:
    def __init__(self, driver_path: str, url: str, username: str, name_xpath: Dict[str, str]):
        self.driver_core: WebDriverCore = WebDriverCore(driver_path)

        self.url: str = url
        self.username: str = username
        self.name_xpath: Dict[str, str] = name_xpath

        self.score: int = 0

        self.start_game()

    def start_game(self):
        """
        1. toggle animation by click
        2. set username by send_keys
        3. press start button by click
        """

        self.driver_core.get(self.url)
        self.driver_core.wait()

        self.driver_core.store_elements('animation_toggle', self.name_xpath['animation_toggle'])
        self.driver_core.click('animation_toggle')

        self.driver_core.store_elements('username_input', self.name_xpath['username_input'])
        self.driver_core.send_text('username_input', self.username)

        self.driver_core.store_elements('start_button', self.name_xpath['start_button'])
        self.driver_core.click('start_button')
        self.driver_core.wait()

        self.driver_core.store_elements('game_canvas', self.name_xpath['game_canvas'])

    def quit(self):
        self.driver_core.quit()

    def get_tile_matrix(self) -> List[List[Tuple[int]]]:
        img = Image.open(BytesIO(self.driver_core.get_png_from('game_canvas')))

        matrix: List[List[Tuple[int]]] = []
        for row_idx in range(15):
            row = []

            for col_idx in range(8):
                pixel_pos = (col_idx * 30 + 15), ((14 - row_idx) * 30 + 15)
                row.append(img.getpixel(pixel_pos)[:3])

            matrix.append(row)

        return matrix

    def poptile(self, pos: Tuple[int, int]):
        x, y = pos
        offset: Tuple[int, int] = (x * 30 + 15), ((14 - y) * 30 + 15)
        self.driver_core.click_with_offset('game_canvas', offset)
        self.driver_core.wait()

    def is_gameover(self) -> bool:
        return self.driver_core.get_url() == 'http://s0af.panty.run/single/result'
