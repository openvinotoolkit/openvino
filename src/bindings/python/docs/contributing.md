# Contributing to OpenVINO:tm: Python API

#### Prerequisites
*To be added...*

##### Enviroment
In case the Python version you have is not supported by OpenVINO, you can refer to [openvino/src/bindings/python/docs/python_version_upgrade.md](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/docs/python_version_upgrade.md) for instructions on how to download and build a newer, supported Python version.
<!-- TODO: Link to enviroment setup -->
*To be added...*

##### Building
<!-- TODO: Link to building instructions -->
*To be added...*

## Contribution guidelines and best practices

#### How to contribute to Python API?
It is nothing special... :) First, make sure that all prerequisites are met and focus on writing the code itself. A good starting point is to have some knowledge of the Python language. C++ is also a vital language for OpenVINO:tm:, so it is not a surprise that it is used in this part of the project as well.

Code snippets and detailed explanations can be found here:
<!-- Link to EXAMPLES -->
    openvino/src/bindings/python/docs/code_example.md

##### Always test out our code! Don't forget about it before pushing and triggering CIs.

Please refer to Test Guide available here:

    openvino/src/bindings/python/docs/test_examples.md

Moreover, the project utilizes *flake8* and *mypy* packages to run codestyle checks. Additionally OpenVINO:tm: uses the custom configuration file to exclude some strict rules. To run codestyle checks, navigate to the main Python API folder first and use following commands:
```shell
cd .../openvino/src/bindings/python/

flake8 src/openvino/ --config=setup.cfg
flake8 tests/ --config=setup.cfg

mypy src/openvino/ --config-file ./setup.cfg
```

**Python API CIs are composed of both functional tests and codestyle checks and may fail because of warnings/errors in both stages.**

##### Adding dependencies to the project
Remember that a new module/feature may be dependent on various third party modules. Please add a sufficient `requirements.txt` file and mention it in a Pull Request description. Consider other project requirements and dependencies of your module to make `requirements.txt` as compact as possible.

**Please look for current supported Python versions and check if packages are compatibile with them and not depreacated.**

#### Description of the Pull Request
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
