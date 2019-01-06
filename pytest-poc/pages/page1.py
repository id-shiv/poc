from selenium import webdriver


class Page1:
    def __init__(self):
        self.driver = None
        self.__base_url = 'https://www.orangehrm.com'
        self.__link_text1 = 'FREE 30 Day Trial'
        self.test_case = None

    def launch(self):
        self.driver = webdriver.Chrome(
            executable_path='/Users/shiv/Desktop/Scripts/poc-scripts/pytest-poc/utilities/webdrivers/chromedriver')
        self.driver.implicitly_wait(10)
        self.driver.maximize_window()
        self.driver.get(self.__base_url)

    def click_link_text1(self):
        self.driver.find_element_by_link_text(self.__link_text1).click()

    def clean_up(self):
        self.driver.close()
        self.driver.quit()

