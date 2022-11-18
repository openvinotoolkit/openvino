# OpenVINO Python API

OpenVINO Python API is a part of OpenVINO library. The component is responsible for:

* Bindings of OpenVINO - allowing users to utilize OpenVINO library in their Python code. Python API provides bindings to basic and advanced APIs from OpenVINO Runtime.
* Extending OpenVINO with pythonic features - on top of direct translations from C++, Python API component:
    * Adds specific extensions to support numpy-based data.
    * Provides support for external frameworks inputs.
    * Provides shortcuts and helpers with more pythonic design.
    * Allows to apply advanced concepts like shared memory to take full advantage of OpenVINO.

## Key contacts

If you have any questions, feature requests or want us to review your PRs, send us a message or ping us on Github.

Team members and main responsibilities:

* [Jan Iwaszkiewicz](https://github.com/jiwaszki) (e-mail: jan.iwaszkiewicz@intel.com)  
Techincal Lead: questions, issues and feature requests, memory and GIL related questions
* [Anastasia Kuporosova](https://github.com/akuporos) (e-mail: anastasia.kuporosova@intel.com)  
Developer: building of the component, general bindings
* [Przemyslaw Wysocki](https://github.com/p-wysocki) (e-mail: przemyslaw.wysocki@intel.com)  
Developer: project requirements, general bindings

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

If you want to contribute to OpenVINO Python API, here is the list of learning materials and project's guidelines:

* [How to contribute](docs/contributing.md)
* [How to extend OpenVINO Python API](docs/code_examples.md)
* [How to test OpenVINO Python API](docs/test_examples.md)
* [How to upgrade local Python version](docs/python_version_upgrade.md)

## See also

Related pages and articles:
* [pybind11 repository](https://github.com/pybind/pybind11)
* [pybind11 documentation](https://pybind11.readthedocs.io/en/stable/)
