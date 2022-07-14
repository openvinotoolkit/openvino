# How to test OpenVINO:tm: Python API?
<!-- Do we keep it as is? -->

#### Prerequisites
<!-- Do we keep it here? -->
*To be added...*

##### Enviroment
<!-- Link to enviroment setup -->
*To be added...*

##### Building
<!-- Link to building instructions -->
*To be added...*

### Running OpenVINO:tm: Python API tests
*For simplicity all of these commands require to navigate to main Python API folder first:*
```
cd .../openvino/src/bindings/python/
```

To run **all tests**:
```
pytest tests/
```

Test framework *pytest* allows to filter tests with `-k` flag.
```
pytest tests/test_inference_engine/test_core.py -k "test_available_devices"
```

Alternatively full name and path to the test case could be passed.
```
pytest tests/test_inference_engine/test_core.py::test_available_devices
```

To print test names and increase verbosity, use `-v` flag.
```
pytest tests/test_inference_engine/test_core.py -v
```
*Tip: look at pytest's documentation for more useful tricks: <!-- Link to it -->*

To run full test suite one can utilize `tox` command:
```
tox
```

### Writing OpenVINO:tm: Python API tests
###### Before start
Follow and complete `openvino/src/bindings/python/docs/code_examples.md`.

##### Adding new test-case in the correct place
Let's add new test for OpenVINO:tm: Python API.

First test should confirm that new pybind11-based class `MyTensor` is behaving correctly. Navigate to tests folder and create new file that describes tests within it. It should be along the lines of:

    tests/test_inference_engine/test_mytensor.py


**Don't forget to include license on the top of each new file!**

Note that name of the file is connected to the class/module to be tested. This is exactly why tests are structured in folders that are describing what tests are supposed to be there. Always add tests to correct places, new folders and files should be created only when necessary. Quick overview of the structure:

    tests/test_frontend           <-- frontend manager and extensions
    tests/test_inference_engine   <-- runtime classes such as Core and Tensor
    tests/test_graph              <-- operators and their implementation
    tests/test_onnx               <-- ONNX Frontend tests and validation
    tests/test_transformations    <-- optimization passes for OV Models 

##### Writing test itself
Let's add a test case for new class. Start with imports and simple test of the creation of a class:
```
import pytest
import numpy as np 
import openvino.runtime as ov

def test_mytensor_creation():
    tensor = ov.MyTensor([1, 2, 3])

    assert tensor is not None
```

Rebuilding step is not necessary here as long as there are no updates to codebase itself. Run the test with:
```
pytest tests/test_inference_engine/test_mytensor.py -v
```

In actual tests it is a good pratice to parametrize them, thus making tests compact and reducing number of handwritten test cases. Additionally, adding checks for shared functions to the basic tests is a common technique. Let's replace the test with:
```
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
```
tests/test_inference_engine/test_mytensor.py::test_mytensor_creation[source0] PASSED                                                                                                                                    [ 50%]
tests/test_inference_engine/test_mytensor.py::test_mytensor_creation[source1] PASSED                                                                                                                                    [100%]
```

Notice that test name is shared between cases. In real-life pull request, all of the functionalities should be tested to ensure the quality of solution. Always focus on general usage and edge-case scenarios. Remember, on the other hand, excessive testing is also not advised as it may result in duplicate test cases and taking a toll on validation pipelines. Good "rule-of-thumb"s while adding tests to the project:
* Don't test built-in capabilities of a given language.
* Common functions can be tested together.
* Create test cases with few standard scenarios and cover all known edge-cases.  
* Hardcode desired results...
* ... or create reference values during runtime. Always use a good, thrust-worthy library for that!
* Re-use common parts of the code (like multiple lines that create helper object) and move them out to make tests easier to read.
* My personal advice: take a little break and start writing tests with fresh mind, try to take a user perspective.

###### Difference between *tests* and *tests_compatibility* directories
<!-- TO-DELETE when compatibility layer is no longer supported in the project -->
Someone could notice two similar folders `tests` and `tests_compatibility`. First one is the desired place for all upcoming features and tests. Compatibility layer is only supported in specific cases and any updates to it should be explicitly approved by OpenVINO:tm: reviewers. Please do not duplicate tests in both directories if not necessary.
