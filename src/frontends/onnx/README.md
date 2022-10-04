# ONNX FE

The main responsibility of ONNX Frontend is import of ONNX models and conversion of these into the `ov::Model` representation. 
Other capabilities of the ONNX Frontend:
* modification of tensors properties (like data type and shapes)
* changing topology of the models (like cutting subgraphs, inserting additional inputs and outputs)
* user-friendly searching the models (via tensors and operators names)

The component is written in `C++`, `Python` bindings are also available.
Each change to ONNX FE component requires `clang-format` code style and `NCC` naming style checks.


## Key contacts

In case of any questions, review and merge requests, please contact:
* Tomasz Dołbniak (tomdol)
* Mateusz Tabaka (mateusztabaka)
* Mateusz Bencer (mbencer)

from `openvino-onnx-frontend-maintainers` maintainers group.


## Components

ONNX Frontend implements an interface common for all frontends defined in [Frontends API](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/common/include/openvino/frontend).
The exported symbols are decorated with `ONNX_FRONTEND_API` macro.
For backward compatibility reasons, the ONNX importer API (more lower-level abstraction approach) is still maintained. It can be found in (ONNX Importer)[https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/onnx/frontend/include/onnx_import/onnx.hpp]. The symbols of ONNX Importer API are decorated with `ONNX_IMPORTER_API` macro.

The crucial place in ONNX Frontend is the directory where the operators are implemented:[https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/onnx/frontend/src/op]. Each operator handler has to be registered in ops bridge - (ops bridge)[https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/onnx/frontend/src/ops_bridge.cpp]. Expect that ONNX frontend has capabilities to register a custom op by a user. It can be achieved via `ConversionExtension` available in (Conversion)[https://github.com/openvinotoolkit/openvino/blob/master/src/frontends/onnx/frontend/include/openvino/frontend/onnx/extension/conversion.hpp].

API of ONNX Frontend can be called directly, but it is also used internally by (Model Optimizer)[https://github.com/openvinotoolkit/openvino/tree/master/tools/mo] during ONNX to IR (Intermediate Representation) conversion. What's more capabilities of ONNX Frontend are used by ONNX Runtime via OpenVINO Execution Provider (more information can be found in (docs)[https://onnxruntime.ai/docs/build/eps.html#openvino]).

ONNX Frontend is tested in few places:
- [C++ gtest-based tests](https://github.com/openvinotoolkit/openvino/tree/master/src/frontends/onnx/tests)
- [Python frontend tests](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python/tests/test_frontend)
- [Python operators tests](https://github.com/openvinotoolkit/openvino/tree/master/src/bindings/python/tests/test_onnx)
- [Python compliance with ONNX standard tests](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/tests/test_onnx/test_backend.py)
- [Python OpenModelZoo tests](https://github.com/openvinotoolkit/openvino/blob/master/src/bindings/python/tests/test_onnx/test_zoo_models.py)


## Architecture
The overview of components responsible for importing scenario is shown on the diagram below:
![ONNX overview diagram](docs/img/onnx_fe_overview.png)


## Tutorials
TBD


## See also

 * [ONNX standard repository](https://github.com/onnx/onnx)
 * [ONNX operators list](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
 * [ONNX Runtime OpenVINO Provider](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/providers/openvino)
