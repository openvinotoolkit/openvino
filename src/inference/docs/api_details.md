# OpenVINO Inference API

OpenVINO Inference API contains two folders:
 * [ie](../include/ie/) - is a legacy API, this API is no longer being developed.
 * [openvino](../include/openvino/) - current public API, this part is described below.

## Components of public OpenVINO Inference API

Public OpenVINO Inference API defines global header [openvino/openvino.hpp](../include/openvino/openvino.hpp) what includes all common OpenVINO headers. 
All Inference components are concentrated inside the [openvino/runtime](../include/openvino/runtime) folder.

To get more information about using of Inference API please read [How to integrate OpenVINO with your application](https://docs.openvino.ai/nightly/openvino_docs_OV_UG_Integrate_OV_with_your_application.html).
Also the diagram of dependencies is presented on the [OpenVINO Architecture page](../../docs/architecture.md#openvino-inference-pipeline).

## Components of OpenVINO Developer API

OpenVINO Developer API is needed for OpenVINO plugin development. This process is detailed explained in the [OpenVINO Plugin Development Guide](https://docs.openvino.ai/nightly/openvino_docs_ie_plugin_dg_overview.html).

## See also
 * [OpenVINO™ Core README](../README.md)
 * [OpenVINO™ README](../../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)

