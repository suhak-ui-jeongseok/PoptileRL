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
            )
            start_element.click()
            break
        except Exception:
            pass

    while True:
        try:
            canvas_element = driver.find_element(
                by=By.XPATH,
                value='//*[@id="cvs"]'
            )
            x, y = map(int, input('x, y > ').split(' '))
            offset = (x * 31 + 15), ((14 - y) * 31 + 15)
            clickTile = action_chains.ActionChains(driver)\
                .move_to_element_with_offset(canvas_element, *offset)\
                .click()

            clickTile.perform()

            print(offset, 'clicked')
        except Exception:
            pass

    input()

    driver.close()