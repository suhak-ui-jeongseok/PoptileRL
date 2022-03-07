import base64
import io
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome import service
from selenium.webdriver.common import action_chains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


if __name__ == '__main__':
    CHROMEDRIVER_PATH = 'YOUR DRIVER PATH'
    service = service.Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service)
    driver.get(url='http://s0af.panty.run/')

    # Disable animation, Set username, click start button
    try:
        animation_toggle_selector = (
            By.XPATH,
            '//*[@id="root"]/div/div/div[1]/button'
        )
        animation_toggle_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(animation_toggle_selector)
        )
        animation_toggle_element.click()

        username_text_selector = (
            By.XPATH,
            '//*[@id="root"]/div/div/input'
        )
        username_text_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(username_text_selector)
        )
        username_text_element.send_keys('Test')

        start_btn_selector = (
            By.XPATH,
            '//*[@id="root"]/div/div/div[2]/a[1]/button'
        )
        start_btn_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(start_btn_selector)
        )
        start_btn_element.click()

    except Exception as e:
        print(e)
    
    input()

    try:
        canvas_selector = (
            By.XPATH,
            '//*[@id="root"]/div/div/canvas'
        )
        canvas_element = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(canvas_selector)
        )

        while True:
            b64_canvas = canvas_element.screenshot_as_base64

            img = Image.open(io.BytesIO(base64.b64decode(b64_canvas)))
            for y in range(15):
                for x in range(8):
                    pixel_pos = (x * 30 + 15), ((14 - y) * 30 + 15)
                    print(img.getpixel(pixel_pos))

            x, y = map(int, input('x, y > ').split(' '))
            if x == -1 and y == -1:
                break
            offset = (x * 31 + 15), ((14 - y) * 31 + 15)

            clickTile = action_chains.ActionChains(driver)\
                .move_to_element_with_offset(canvas_element, *offset)\
                .click_and_hold()\
                .pause(0.05)\
                .release()

            clickTile.perform()

            print(offset, 'clicked')
    except Exception as e:
        print(e)

    input()

    driver.close()
