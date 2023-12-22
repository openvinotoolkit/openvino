# How to test OpenVINO™ Python API?

## Building and environment
Instructions can be found in ["Building the OpenVINO™ Python API"](./build.md).

Install the specific requirements file for testing:
```
python -m pip install -r openvino/src/bindings/python/requirements_test.txt
```

Make sure that Python libraries are added to the user environment variables: 
```
export PYTHONPATH=PYTHONPATH:<openvino_repo>/bin/intel64/Release/python
```
## Run OpenVINO™ Python API tests
*For simplicity, all of these commands require to navigate to the [main Python API folder](./../) first:*
```shell
cd .../openvino/src/bindings/python/
```

To run OpenVINO Python API 2.0 tests:
```shell
pytest tests/
```

To run OpenVINO Python API 1.0 tests, use this command:
```
pytest tests_compatibility/
```

By default, tests are run on the CPU plugin. If you want to run them on a different plugin,
you need to specify this environment variable:
```
export TEST_DEVICE=GPU
```

The *pytest* test framework enables you to filter tests with the `-k` flag.
```shell
pytest tests/test_runtime/test_core.py -k "test_available_devices"
```

Alternatively, the full name and path to the test case could be passed.
```shell
pytest tests/test_runtime/test_core.py::test_available_devices
```

To print test names and increase verbosity, use `-v` flag.
```shell
pytest tests/test_runtime/test_core.py -v
```
*Tip: look at pytest's documentation for more useful tricks: https://docs.pytest.org/en/latest/*

To run full test suite one can utilize `tox` command:
```shell
tox
```

## Check the codestyle of Python API
There are two packages used in the project to check the codestyle of python code: *mypy* and *flake8*.
Besides, OpenVINO™ uses a [custom configuration file](./../setup.cfg) to exclude some strict rules.

To check the codestyle of the Python API 2.0, run the following commands:
```
python -m flake8 ./src/openvino/ --config=setup.cfg
python -m mypy ./src/openvino --config-file ./setup.cfg
```
To check the codestyle of the nGraph Python API, run the following commands:
```
python -m flake8 ./src/compatibility/ngraph/ --config=setup.cfg
python -m mypy ./src/compatibility/ngraph --config-file ./setup.cfg
```
To check the codestyle of the InferenceEngine Python API, run the following commands:
```
cd src/compatibility/openvino
python -m flake8 ./ --config=setup.cfg
python -m mypy ./ --config-file ./setup.cfg
```
It's recommended to run the mentioned codestyle check whenever new tests are added.
This check should be executed from the main Python API folder:
```
python -m flake8 ./tests/ --config=setup.cfg
```
## Writing OpenVINO™ Python API tests
### Before start
Follow and complete [Examples of OpenVINO™ Python API code](./code_examples.md).

### Adding new test-case in the correct place
Let's add a new test for OpenVINO™ Python API.

First, the test should confirm that the new pybind11-based class of `MyTensor` is behaving correctly. Navigate to [tests folder](./../tests/test_runtime/) and create a new file `test_mytensor.py` that describes tests within it. Final path should be along the lines of:

    tests/test_runtime/test_mytensor.py


**Don't forget to include license on the top of each new file!**

Note that name of the file is connected to the class/module to be tested. This is exactly why tests are structured in folders that are describing what tests are supposed to be there. Always add tests to correct places, new folders and files should be created only when necessary. Quick overview of the structure:

    tests/test_frontend           <-- frontend manager and extensions
    tests/test_runtime            <-- runtime classes such as Core and Tensor
    tests/test_graph              <-- operators and their implementation
    tests/test_onnx               <-- ONNX Frontend tests and validation
    tests/test_transformations    <-- optimization passes for OV Models 

### Writing of the test itself
Let's add a test case for new class. Start with imports and simple test of the creation of a class:
```python
import pytest
import numpy as np 
import openvino as ov

def test_mytensor_creation():
    tensor = ov.MyTensor([1, 2, 3])

    assert tensor is not None
```

Rebuilding step is not necessary here as long as there are no updates to codebase itself. Run the test with:
```shell
pytest tests/test_runtime/test_mytensor.py -v
```

In actual tests it is a good pratice to parametrize them, thus making tests compact and reducing number of handwritten test cases. Additionally, adding checks for shared functions to the basic tests is a common technique. Let's replace the test with:
```python
@pytest.mark.parametrize(("source"), [
    ([1, 2, 3]),
    (ov.Tensor(np.array([4, 5 ,6]).astype(np.float32))),
])
def test_mytensor_creation(source):
    tensor = ov.MyTensor(source)

    assert tensor is not None
    assert tensor.get_size() == 3
```

Run the tests, output should be similar to:
```shell
tests/test_runtime/test_mytensor.py::test_mytensor_creation[source0] PASSED                                                                                                                                    [ 50%]
tests/test_runtime/test_mytensor.py::test_mytensor_creation[source1] PASSED                                                                                                                                    [100%]
```

Notice that the test name is shared between cases. In a real-life pull request, all of the functionalities should be tested to ensure the quality of the solution. Always focus on general usage and edge-case scenarios. On the other hand, remember that excessive testing is not advised as it may result in duplicate test cases and impact validation pipelines. A good "rule-of-thumb" list of practices while adding tests to the project is:
* Don't test built-in capabilities of a given language.
* Common functions can be tested together.
* Create test cases with a few standard scenarios and cover all known edge-cases.  
* Hardcode desired results...
* ... or create reference values during runtime. Always use a good, thrust-worthy library for that!
* Re-use common parts of the code (like multiple lines that create helper object) and move them out to make tests easier to read.

### Difference between *tests* and *tests_compatibility* directories
<!-- TO-DELETE when compatibility layer is no longer supported in the project -->
Someone could notice two similar folders [`tests`](./../tests/) and [`tests_compatibility`](./../tests_compatibility/). First one is the desired place for all upcoming features and tests. Compatibility layer is only supported in specific cases and any updates to it should be explicitly approved by OpenVINO™ reviewers. Please do not duplicate tests in both directories if not necessary.

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
