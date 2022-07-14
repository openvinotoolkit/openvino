# Contributing to OpenVINO:tm: Python API
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

## Contribution guidelines and best practices
<!-- Do we keep it as is? -->

#### How to contribute to Python API?
It is nothing special... :) First make sure that all prerequisites are met and focus on writing code itself. Good starting point is to have some knowledge of Python language. C++ is also a vital language for the OpenVINO:tm:, so it is not a surprise that it is used in this part of the project as well.

Code snippets and detailed explanations can be found here:
<!-- Link to EXAMPLES -->
    openvino/src/bindings/python/docs/code_example.md

##### Always test out our code! Don't forget about it before pushing and triggering CIs.

Please refer to Test Guide available here:

    openvino/src/bindings/python/docs/test_examples.md

Moreover, the project utilizes *flake8* and *mypy* packages to run codestyle checks. Additionally OpenVINO:tm: uses the custom configuration file to exclude some strict rules. To run codestyle checks, navigate to main Python API folder first and use following commands:
```
cd .../openvino/src/bindings/python/

flake8 src/openvino/ --config=setup.cfg
flake8 tests/ --config=setup.cfg

mypy src/openvino/ --config-file ./setup.cfg
```

**Python API CIs are composed of both functional tests and codestyle checks and may fail because of warnings/errors in both stages.**

##### Adding dependencies to the project
Remember that new module/feature may be dependant on various third party modules. Please add sufficent `requirements.txt` file and mention it in Pull Request description. Consider other project requirements and dependencies of your module to make `requirements.txt` as compact as possible.

**Please look for current supported Python versions and check if packages are compatibile with them and not depreacated.**

#### Description of the Pull Request
Please append all PR titles with a tag `[PyOV]` or `[PYTHON]`. Feel free to describe any level of relevant details in PR, it helps a lot during reviewing process. A minimum requirement is compact description of made changes, form of a bullet-point list is really appreciated.

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
