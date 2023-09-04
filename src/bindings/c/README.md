# OpenVINO C API

OpenVINO C API is a key part of the OpenVINO extension for C API users. This component provides C API for OpenVINO Toolkit.

```mermaid
flowchart LR
    c_application[("C Application")]
    openvino_c{{openvino::c}}
    openvino{{openvino}}

    style c_application fill:#427cb0
    style openvino_c fill:#6c9f7f
    style openvino fill:#6c9f7f

    c_application-->openvino_c-->openvino
```

OpenVINO C API uses [the common coding style rules](../../../docs/dev/coding_style.md).

## Key contacts

People from the [openvino-c-api-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-c-api-maintainers) group have the rights to approve and merge PRs to the C API component. They can assist with any questions about C API component.

## Components

OpenVINO C API has the following structure:
 * [docs](./docs) contains developer documentation for OpenVINO C APIs.
 * [include](./include) contains all provided C API headers. [Learn more](https://docs.openvino.ai/2023.1/api/api_reference.html).
 * [src](./src) contains the implementations of all C APIs.
 * [tests](./tests) contains all tests for OpenVINO C APIs. [Learn more](./docs/how_to_write_unit_test.md).

> **NOTE**:  Using API 2.0 is strongly recommended. Legacy API (for C) [header file](./include/c_api/ie_c_api.h), [source file](./src/ie_c_api.cpp), [unit test](./tests/ie_c_api_test.cpp) are also included in the component, but the legacy API is no longer extended. 

## Tutorials

* [How to integrate OpenVINO C API with Your Application](https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_Integrate_OV_with_your_application.html)
* [How to wrap OpenVINO objects with C](./docs/how_to_wrap_openvino_objects_with_c.md)
* [How to wrap OpenVINO interfaces with C](./docs/how_to_wrap_openvino_interfaces_with_c.md)
* [Samples implemented by OpenVINO C API](../../../samples/c/)
* [How to debug C API issues](./docs/how_to_debug_c_api_issues.md)
* [How to write unit test](./docs/how_to_write_unit_test.md)

## How to contribute to the OpenVINO repository

See [CONTRIBUTING](../../../CONTRIBUTING.md) for details.

## See also

 * [OpenVINO™ README](../../../README.md)
 * [OpenVINO Runtime C API User Guide](https://docs.openvino.ai/2023.1/openvino_docs_OV_UG_Integrate_OV_with_your_application.html)
 * [Migration of OpenVINO C API](https://docs.openvino.ai/2023.1/openvino_2_0_transition_guide.html)
