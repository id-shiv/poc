"""
Name of the python file is prefixed with test to the name of python file that
contains modules
"""

import sys
sys.path.insert(0, '/Users/shiv/Desktop/Scripts/poc-scripts/pytest-poc/pages')
from page1 import Page1
page = Page1()


def test_test_case1():
    page.launch()
    page.click_link_text1()


def test_test_case2():
    page.launch()
    page.click_link_text1()


def teardown():
    page.clean_up()
