# OpenVINO C API

OpenVINO C API is a part of OpenVINO Runtime library. 
This component is responsible for:
 * Based representations for the objects in OpenVINO - C API provides structures with C code for basical classes inside the OpenVINO, such as "ov_core_t" for "ov::Core". For more information please read [OpenVINO Runtime C API User Guide](./docs/OpenVINO_Runtime_C_API_User_Guide.md).
 * Based operations representation from OpenVINO - C API contains basical interfaces (read model, inference, properity set and so on) with C code. For more information please read [OpenVINO Runtime C API User Guide](./docs/OpenVINO_Runtime_C_API_User_Guide.md).

OpenVINO C API uses [the common coding style rules](../../docs/dev/coding_style.md).

## Key person

People from the [openvino-c-api-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-c-api-maintainers) allows to approve and merge PRs to the C API component. These guys can help in case of any questions about C API component.

## Components

OpenVINO C API has the next structure:
 * [docs](./docs) contains developer documentation pages for OpenVINO Runtime C API.
 * [include](./include) contains public API. Details information about provided API can be found [here](./docs/OpenVINO_Runtime_C_API_User_Guide.md).
 * [src](./src) folder contains sources of all C API implementation.
 * [tests](./tests) contains tests for OpenVINO C API components. More information about OpenVINO Core tests can be found [here](./docs/tests.md).

## Tutorials
- [Hello Classification C Sample](../../../samples/c/hello_classification/README.md) Inference of image classification networks like AlexNet and GoogLeNet using Synchronous Inference Request API. Input of any size and layout can be set to an infer request which will be pre-processed automatically during inference (the sample supports only images as inputs and supports Unicode paths).
- [Hello NV12 Input Classification C Sample](../../../samples/c/hello_nv12_input_classification/README.md) Input of any size and layout can be provided to an infer request. The sample transforms the input to the NV12 color format and pre-process it automatically during inference. The sample supports only images as inputs.

## See also
 * [OpenVINO Runtime C API User Guide](./docs/OpenVINO_Runtime_C_API_User_Guide.md)
 * [OpenVINO Runtime C API Developer Guide](./docs/OpenVINO_Runtime_C_API_Developer_Guide.md)
