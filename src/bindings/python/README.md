# OpenVINO Python API

OpenVINO Python API is a part of OpenVINO library. The component is responsible for:

* Bindings of OpenVINO - allowing users to utilize OpenVINO library in their Python code. Python API provides bindings to basic and advanced APIs from OpenVINO Runtime.
* Extending OpenVINO with pythonic features - on top of direct translations from C++, Python API component:
    * Adds specific extensions to support numpy-based data.
    * Provides support for external frameworks inputs.
    * Provides shortcuts and helpers with more pythonic design.
    * Allows to apply advanced concepts like shared memory to take full advantage of OpenVINO.

OpenVINO Python API uses [common coding style rules](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/docs/contributing.md#contribution-guidelines-and-best-practices) that are adjusted to project needs.

## Key contacts

If you have any questions, feature requests or want us to review your PRs, send us a message or ping us on GitHub via [openvino-ie-python-api-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-python-api-maintainers). You can always directly contact everyone from this group.

## Components

```
openvino/src/bindings/python
├── docs                    <-- Developer documentation
│   └── examples            <-- Developer code examples
├── README.md               <-- This file:)
├── src
│   ├── compatibility       <-- Sources for compatibility API
│   ├── openvino            <-- Python sources for current API
│   └── pyopenvino          <-- C++ sources for current API
├── tests
├── tests_compatibility
├── thirdparty              <-- Thirdparty libraries
│   └── pybind11
└── wheel                   <-- Wheel specific directory
```

## Tutorials

If you want to contribute to OpenVINO Python API, here is the list of learning materials and project guidelines:

* [How to contribute](./docs/contributing.md)
* [How to extend OpenVINO Python API](./docs/code_examples.md)
* [How to test OpenVINO Python API](./docs/test_examples.md)
* [How to upgrade local Python version](./docs/python_version_upgrade.md)

## See also

* [OpenVINO™ README](../../../README.md)
* [OpenVINO™ Core Components](../../README.md)
* [OpenVINO™ Python API Reference](https://docs.openvino.ai/latest/api/ie_python_api/api.html)
* [OpenVINO™ Python API Exclusives](https://docs.openvino.ai/latest/openvino_docs_OV_UG_Python_API_exclusives.html)
* [pybind11 repository](https://github.com/pybind/pybind11)
* [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/)
