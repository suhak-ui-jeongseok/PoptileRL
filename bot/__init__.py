import base64
import io

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome import service
from selenium.webdriver.common import action_chains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from bot_manager import full_routine