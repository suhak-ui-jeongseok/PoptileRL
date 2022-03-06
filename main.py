import base64
import io
from PIL import Image
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome import service as chrome_service
from selenium.webdriver.common import action_chains


if __name__ == '__main__':
    CHROMEDRIVER_PATH = 'YOUR DRIVER PATH'
    service = chrome_service.Service(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service)
    driver.get(url='http://s0af.panty.run/static/poptile/classic')

    while True: 
        try:
            start_element = driver.find_element(
                by=By.XPATH,
                value='//*[@id="app"]/div/div/div[4]/input[2]'
            )  #TODO: fix into selenium waituntil API
            start_element.click()
            break
        except Exception:
            pass

    while True:
        try:
            canvas_element = driver.find_element(
                by=By.XPATH,
                value='//*[@id="cvs"]'
            )  #TODO: fix into selenium waituntil API
            b64_canvas = canvas_element.screenshot_as_base64

            img = Image.open(io.BytesIO(base64.b64decode(b64_canvas)))
            for y in range(15):
                for x in range(8):
                    pixel_pos = (x * 31 + 15), ((14 - y) * 31 + 15)
                    print(img.getpixel(pixel_pos))

            x, y = map(int, input('x, y > ').split(' '))
            if x == -1 and y == -1:
                break
            offset = (x * 31 + 15), ((14 - y) * 31 + 15)
            
            clickTile = action_chains.ActionChains(driver)\
                .move_to_element_with_offset(canvas_element, *offset)\
                .click()

            clickTile.perform()

            print(offset, 'clicked')
        except Exception as e:
            print(e)

    input()

    driver.close()