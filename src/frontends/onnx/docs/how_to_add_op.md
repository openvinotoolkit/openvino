# How to add a new operation

## How to implement a new operation in ONNX FE codebase
ONNX operations ("op" or "ops" for short in this article) can be distinguished into two main categories: [the official ops defined in the ONNX standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md) and the custom-domain ops (such as ops from the `org.openvinotoolkit`, `com.microsoft`, and `org.pytorch.aten` domains). Multiple operator handlers for different versions of an op can be defined. When importing a model, ONNX FE tries to use a handler that matches the version of the opset in the model. If such implementation doesn't exist, it will try to use the existing handler(s) starting with the greatest opset number. When adding a new operator's implementation, the implementation has to be registered using version `1` (according to an implementation requirement of ONNX FE), even if the operation has been added to the ONNX standard in an opset greater than 1.

List of a currently available ops is available [in Supported Operations](supported_ops.md).

For example, we want to implement our new `org.openvinotoolkit.CustomAdd` operation in version `1`.
The first step is to add `.cpp` file in [the ops folder](../../../../src/frontends/onnx/frontend/src/op). For this particular case, it should be [op/org.openvinotoolkit](../../../../src/frontends/onnx/frontend/src/op/org.openvinotoolkit) to be consistent with the op folder layout.

The definition in `.cpp` contains an implementation of transformation from [ov::frontend::onnx::Node](../../../../src/frontends/onnx/frontend/include/onnx_import/core/node.hpp) to [ov::OutputVector](../../../../src/core/include/openvino/core/node_vector.hpp).

Transformation must be implemented in a correct namespace, which is defined as ov::frontend::onnx::domain::opset_N. Domain is operation domain name (ai_onnx for ONNX standard), opset_N - defined operation opset.

The next step is to register a new op by using a macro [ONNX_OP](../../../../src/frontends/onnx/frontend/src/core/operator_set.hpp). For `org.openvinotoolkit.CustomAdd`, the registration can look like:
```cpp
ONNX_OP("CustomAdd", OPSET_SINCE(1), ai_onnx::opset_1::custom_add, OPENVINO_ONNX_DOMAIN);
```

Whole implementation can look like:
```cpp
#include "core/operator_set.hpp"

#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector custom_add(const ov::frontend::onnx::Node& node) {
    // CustomAdd should have at least 2 inputs
    common::default_op_checks(node, 2);

    const auto& inputs = node.get_ov_inputs();
    const auto in1 = inputs[0];
    const auto in2 = inputs[1];
    const auto alpha = node.get_attribute_value<float>("alpha", 1);

    CHECK_VALID_NODE(node,
                     alpha >= 1 && alpha < 100,
                     "CustomAdd accepts alpha in a range [1, 100), got: ",
                     alpha);

    const auto alpha_node =
        std::make_shared<v0::Convert>(v0::Constant::create(ov::element::f32, {}, {alpha}), in1.get_element_type());

    const auto add = std::make_shared<v1::Add>(in1, in2);
    return {std::make_shared<v1::Multiply>(add, alpha_node)};
}

ONNX_OP("CustomAdd", OPSET_SINCE(1), ai_onnx::opset_1::custom_add, OPENVINO_ONNX_DOMAIN);

}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
```

The minimum requirement to receive an approval during the code review is the implementation of [C++ unit tests](tests.md#C++-tests) for a new operation. To make a test you may need to prepare a [prototxt file](../tests/models) with a small model which implements your operation.


## How to register a custom operation via extensions mechanism
The complete tutorial about custom frontends extensions can be found in [frontend extensions](../../../../docs/Extensibility_UG/frontend_extensions.md). The section below will show you the most useful ways of adding extensions for the ONNX Frontend.
### C++ based extensions
To register your ONNX node-OV subgraph mapping, you can use `ConversionExtension` with syntax as below:
```cpp
core.add_extension(ov::frontend::onnx::ConversionExtension("CustomAdd", "org.openvinotoolkit", ov::frontend::CreatorFunction(
                                                            [](const ov::frontend::NodeContext& context)
                                                            {
                                                                const auto add = std::make_shared<ov::opset9::Add>(context.get_input(0), context.get_input(1));
                                                                return add->outputs();
                                                            })));
```
If an OpenVINO Core operation provides exactly what you need (without decomposition to subgraph), `OpExtension` can be a good choice. An example of usage can look like below:
```cpp
core.add_extension(ov::frontend::onnx::OpExtension<ov::opset9::Add>("CustomAdd", "org.openvinotoolkit"));
```
### Python-based extensions
C++ based extensions have their equivalents in Python. For `ConversionExtension`, an example of usage can look like:
```python
from openvino.frontend.onnx import ConversionExtension
...
def custom_add(node: NodeContext):
    input_1 = node.get_input(0)
    input_2 = node.get_input(1)
    add = ops.add(input_1, input_2)
    return [add.output(0)]

fe.add_extension(ConversionExtension("CustomAdd", "org.openvinotoolkit", custom_add))
```
If you use `OpExtension`, an custom op registration can look like:
```python
from openvino.frontend.onnx import OpExtension
...
fe.add_extension(OpExtension("opset9.Add", "CustomAdd", "org.openvinotoolkit", {}, {"auto_broadcast": "numpy"}))
```

## See also
 * [OpenVINO ONNX Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)
