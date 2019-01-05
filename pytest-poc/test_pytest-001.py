"""
Name of the python file is prefixed with test to the name of python file that
contains modules
"""

import pytest_sample1

# Define test functions (starts with prefix test)
def test_add():
    assert pytest_sample1.add(2, 3) == 5
    assert pytest_sample1.add(5) == 7

def test_multiply():
    assert pytest_sample1.multiply(2, 3) == 6
    assert pytest_sample1.multiply(5) == 10
