# Python Selenium | Pytest Framework

trigger script : launcher.py

Pytest
Install Pytest : pip install pytest

Result in verbose
Pytest <script> -v

Run only one module
Pytest <script>::<module_name>

Run modules matching name
Pytest <sctipt> -k “<match_name>”
Pytest <sctipt> -k “<match_name_1> or/add <match_name_2>”

Run marked modules
@pytest.mark.<marker> as module decorator
Pytest <sctipt> -m marker

Exit on first module failure
Pytest <sctipt> -x
Pytest <sctipt> —max-fail=2

—tb=no : Disable trace

Skip a module
@pytest.mark.skip(reason=“do not run”)
@pytest.mark.skipif(sys.version > 3.3, reason=“do not run”)

Print using -s option

-q : for quiet mode

If module needs to be called for multiple inputs, use parameter decorator
@pytest.mark.parameterize(‘arg1’, ‘arg2’, ‘result’, [(7, 3, 10],
    (“test”, “2”, “test2"), ())

Create pre-requisite and cleanup of module using setup and teardown modules

@pytest.fixture - explore this
