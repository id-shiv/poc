import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(sys.path)

os.system('pytest /Users/shiv/Desktop/Scripts/poc-scripts/pytest-poc/tests/test_feature1.py')
