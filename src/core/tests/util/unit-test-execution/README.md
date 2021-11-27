These test executes 'unit-test'.

To run please do the following:
1. To run tests required installing some dependencies:
    - pip3 install -r requirements.txt
2. Set environment variable:
    a. Required:
        export PATH_TO_EXE=<path where nGraph unit-test locates>
3. To run all tests:
    a. cd folder where unit_test_executable.py locates
    b. pytest --gtest_filter="*"
4. To run exact test:
    a. cd folder where unit_test_executable.py locates
    b. pytest --gtest_filter="<your test name>"
5. To get html report add "--html=report.html" to pytest cmd line
    (but before install this module "pip install pytest-html")
6.This test get result of opset1 operation (passed and failed) and also creates csv file 'nodes_coverage.csv' and
    'nodes_coverage.html' after execution. Here you may find name of operations and its passrate and coverage
    for several plugins.
    Example:
    Operation | GPU passed / total | CPU passed / total
    Abs       | 1/2                | 1/2

    Here operation 'Abs': 1 test of 2 passed on GPU and CPU