# Contributing to OpenVINO™ Python API

## Prerequisites

### Enviroment
In case the Python version you have is not supported by OpenVINO, you can refer to ["Python version upgrade" guide](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/docs/python_version_upgrade.md) for instructions on how to download and build a newer, supported Python version.

The environment setup is described as part of the pyenv example in [Building the OpenVINO™ Python API](./build.md#Example:_using_pyenv_with_OpenVINO™_on_Linux_based_system).

### Building
Building instructions can be found in [Building the OpenVINO™ Python API](./build.md#_Building_the_OpenVINO™_Python_API).

## Contribution guidelines and best practices

### How to contribute to Python API?
It is nothing special... :) First, make sure that all prerequisites are met and focus on writing the code itself. A good starting point is to have some knowledge of the Python language. C++ is also a vital language for OpenVINO™, so it is not a surprise that it is used in this part of the project as well.

Code snippets and detailed explanations can be found here:

[Examples of OpenVINO™ Python API code](./code_example.md)

### Always test out our code! Don't forget about it before pushing and triggering CIs.

To learn how to test your code, refer to the guide on [how to test OpenVINO™ Python API?](./test_examples.md#Running_OpenVINO™_Python_API_tests)

Moreover, the project utilizes *flake8* and *mypy* packages to run codestyle checks. Additionally OpenVINO™ uses the custom configuration file to exclude some strict rules. To run codestyle checks, navigate to the main Python API folder first and use following commands:
```shell
cd .../openvino/src/bindings/python/

flake8 src/openvino/ --config=setup.cfg
flake8 tests/ --config=setup.cfg

mypy src/openvino/ --config-file ./setup.cfg
```

**Python API CIs are composed of both functional tests and codestyle checks and may fail because of warnings/errors in both stages.**

### Adding dependencies to the project
Remember that a new module/feature may be dependent on various third party modules. Please add a sufficient `requirements.txt` file and mention it in a Pull Request description. Consider other project requirements and dependencies of your module to make `requirements.txt` as compact as possible.

**Please look for current supported Python versions and check if packages are compatibile with them and not depreacated.**

### Description of the Pull Request
Please append all PR titles with a tag `[PyOV]` or `[PYTHON]`. Feel free to describe any level of relevant details in the PR, it helps a lot with the review process. The minimum requirement is a compact description of changes made, the form of a bullet-point list is really appreciated.

Template for external contributors:
```
Details:
...

Requirements introduced/changed:       <-- only if applicable
...
```

Template for internal contributors:
```
Details:
...

Requirements introduced/changed:       <-- only if applicable
...

Tickets:
XXXX, YYYY                             <-- only numbers from tickets
```

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [OpenVINO™ bindings README](../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
