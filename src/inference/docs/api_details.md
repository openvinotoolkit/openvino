# OpenVINO Inference API

OpenVINO Inference API contains two folders:
 * [openvino](../include/openvino/) - current public API, this part is described below.

## Components of Public OpenVINO Inference API

Public OpenVINO Inference API defines global header [openvino/openvino.hpp](../include/openvino/openvino.hpp) which includes all common OpenVINO headers.
All Inference components are placed inside the [openvino/runtime](../include/openvino/runtime) folder.

To learn more about the Inference API usage, read [How to integrate OpenVINO with your application](https://docs.openvino.ai/2025/openvino-workflow/running-inference.html).
The diagram with dependencies is presented on the [OpenVINO Architecture page](../../docs/architecture.md#openvino-inference-pipeline).

## Components of OpenVINO Developer API

OpenVINO Developer API is required for OpenVINO plugin development. This process is described in the [OpenVINO Plugin Development Guide](https://docs.openvino.ai/2025/documentation/openvino-extensibility/openvino-plugin-library.html).

## See also
 * [OpenVINO™ Core README](../README.md)
 * [OpenVINO™ README](../../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)

