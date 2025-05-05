# OpenVINO Python API

OpenVINO Python API is a part of the OpenVINO library. The component is responsible for:

* Bindings of OpenVINO - allowing users to use the OpenVINO library in their Python code. Python API provides bindings to basic and advanced APIs from OpenVINO Runtime.
* Extending OpenVINO with pythonic features - on top of direct translations from C++, Python API component:
    * Adds specific extensions to support numpy-based data.
    * Provides support for external frameworks inputs.
    * Provides shortcuts and helpers with more pythonic design.
    * Allows to apply advanced concepts, like shared memory, to take full advantage of OpenVINO.

OpenVINO Python API uses [the common codestyle checks](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/docs/contributing.md#contribution-guidelines-and-best-practices) which are adjusted to project needs.

## Key contacts

If you have any questions, feature requests or want us to review your PRs, send us a message or ping us on GitHub via [openvino-ie-python-api-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-python-api-maintainers). You can always directly contact everyone from this group.

## Components

OpenVINO PYTHON API has the following structure:

* [docs](./docs/) - folder that contains developer documentation and code examples.
* [src](./src/) - folder with all source files for Python API.
    * [src/openvino](./src/openvino/) - Python sources.
    * [src/openvino/preprocess](./src/openvino/preprocess/) - Torchvision to OpenVINO preprocessing converter.
    * [src/pyopenvino](./src/pyopenvino/) - C++ sources.
* [tests](./tests/) - tests directory for OpenVINO Python API.
* [thirdparty](./thirdparty/) - folder that contains third-party modules like `pybind11`.
* [wheel](./wheel/) - wheel-specific directory that contains all specific requirements and files used during wheel creation.

## Tutorials

If you want to contribute to OpenVINO Python API, here is the list of learning materials and project guidelines:

* [How to contribute](./docs/contributing.md)
* [How to extend OpenVINO Python API](./docs/code_examples.md)
* [How to test OpenVINO Python API](./docs/test_examples.md)
* [How to upgrade local Python version](./docs/python_version_upgrade.md)
* [How we manage stub .pyi files](./docs/stubs.md)

## See also

* [OpenVINO™ README](../../../README.md)
* [OpenVINO™ Core Components](../../README.md)
* [OpenVINO™ Python API Reference](https://docs.openvino.ai/2025/api/ie_python_api/api.html)
* [OpenVINO™ Python API Advanced Inference](https://docs.openvino.ai/2025/openvino-workflow/running-inference/python-api-advanced-inference.html)
* [OpenVINO™ Python API Exclusives](https://docs.openvino.ai/2025/openvino-workflow/running-inference/python-api-exclusives.html)
* [pybind11 repository](https://github.com/pybind/pybind11)
* [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/)
