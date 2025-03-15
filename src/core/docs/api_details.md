# OpenVINO Core API

OpenVINO Core API contains two folders:
 * [openvino](../include/openvino/) - current public API, this part is described below.

## Structure of Core API
<pre>
 <code>
 <a href="../include/openvino">openvino/</a>                  // Common folder with OpenVINO API
    <a href="../include/openvino/core/">core/</a>             // Contains common classes which are responsible for model representation
    <a href="../include/openvino/op/">op/</a>                 // Contains all supported OpenVINO operations
    <a href="../include/openvino/opsets/">opsets/</a>         // Contains definitions of each official OpenVINO opset
    <a href="../include/openvino/pass/">pass/</a>             // Defines classes for developing transformation and several common transformations
    <a href="../include/openvino/runtime/">runtime/</a>       // Contains OpenVINO tensor definition
 </code>
</pre>

## Main structures for model representation

* `ov::Model` is located in [openvino/core/model.hpp](../include/openvino/core/model.hpp) and provides API for model representation. For more details, read [OpenVINO Model Representation Guide](https://docs.openvino.ai/2025/openvino-workflow/running-inference/model-representation.html).
* `ov::Node` is a base class for all OpenVINO operations, the class is located in the [openvino/core/node.hpp](../include/openvino/core/node.hpp).
* `ov::Shape` and `ov::PartialShape` classes represent shapes in OpenVINO, these classes are located in the [openvino/core/shape.hpp](../include/openvino/core/shape.hpp) and [openvino/core/partial_shape.hpp](../include/openvino/core/partial_shape.hpp) respectively. For more information, read [OpenVINO Shapes representation](./shape_propagation.md#openvino-shapes-representation).
* `ov::element::Type` class represents element type for OpenVINO Tensors and Operations. The class is located in the [openvino/core/type/element_type.hpp](../include/openvino/core/type/element_type.hpp).
* `ov::Tensor` is used for memory representation inside OpenVINO. The class is located in the [openvino/runtime/tensor.hpp](../include/openvino/runtime/tensor.hpp).

## See also
 * [OpenVINO™ Core README](../README.md)
 * [OpenVINO™ README](../../../README.md)
 * [Developer documentation](../../../docs/dev/index.md)
