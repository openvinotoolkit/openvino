# How to add a new operation

## How to implement a new operation
Two main categories of ONNX operations can be distinguished in ONNX Frontend: [official ops defined in ONNX standard](https://github.com/onnx/onnx/blob/main/docs/Operators.md) and custom ops (like ops from `org.openvinotoolkit`, `com.microsoft` or `org.pytorch.aten` domains). Each operation can be defined in one or more versions. Note that ONNX Frontend try to use version of operation defined in the opset of a model at first. If such implementation doesn't exist, the fallback to the most previous existed version is called. What's more the new operation which has only one implementation has to be registered in version "1".

Let's say, that we want to implement our new `org.openvinotoolkit.CustomAdd` operation in version "1".
The first step should be adding `.cpp` and `.hpp` files in [op folder](../../../../src/frontends/onnx/frontend/src/op) (for this particular case it should be [op/org.openvinotoolkit](../../../../src/frontends/onnx/frontend/src/op/org.openvinotoolkit) to be consistent with op folder layout).
The declaration in `.hpp` can look like:
```
#pragma once

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector custom_add(const Node& node);

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
```
The definition in `.cpp` contains implementation of transformation from [ngraph::onnx_import::Node](../../../../src/frontends/onnx/frontend/include/onnx_import/core/node.hpp) to [ov::OutputVector](../../../../src/core/include/openvino/core/node_vector.hpp). Such implementation can look like:
```
#include "op/org.openvinotoolkit/custom_add.hpp"

#include <memory>

#include "default_opset.hpp"
#include "utils/common.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector custom_add(const Node& node) {
    const auto in1 = node.get_ng_inputs().at(0);
    const auto in2 = node.get_ng_inputs().at(1);
    const auto alpha = node.get_attribute_value<float>("alpha", 1);
    const auto alpha_node =
        std::make_shared<default_opset::Convert>(default_opset::Constant::create(element::f32, {}, {alpha}),
                                                 in1.get_element_type());

    const auto add = std::make_shared<default_opset::Add>(in1, in2);
    return {std::make_shared<default_opset::Multiply>(add, alpha_node)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
```
The next step is registration new op in [ops_bridge](../../../../src/frontends/onnx/frontend/src/ops_bridge.cpp). For `org.openvinotoolkit.CustomAdd` the registration can look like:
```
#include "op/org.openvinotoolkit/custom_add.hpp"
...
REGISTER_OPERATOR_WITH_DOMAIN(OPENVINO_ONNX_DOMAIN, "CustomAdd", 1, custom_add);
```
The minimum requirement to receive approval during code review is implementation [C++ unit tests](tests.md#C++-tests).


## How to register a custom operation via Extension mechanism
The complete tutorial about custom frontends extensions can be found in [frontend extensions](../../../../docs/Extensibility_UG/frontend_extensions.md). In the section below are shown the most useful way of adding extensions for ONNX Frontend.
### C++ based extensions
If you need to register your ONNX node-OV subgraph mapping, you can use `ConversionExtension` with syntax like below:
```
core.add_extension(ov::frontend::onnx::ConversionExtension("org.openvinotoolkit", "CustomAdd", ov::frontend::CreatorFunction(
                                                            [](const ov::frontend::NodeContext& context)
                                                            {
                                                                const auto add = std::make_shared<ov::opset9::Add>(context.get_input(0), context.get_input(1));
                                                                return add->outputs();
                                                            })));
```
If OV Core operation provides exactly what you need (without additional transformations) `OpExtension` should be used with syntax like below:
```
core.add_extension(ov::frontend::onnx::OpExtension<ov::opset9::Add>("org.openvinotoolkit", "CustomAdd"));
```
If you need to register a custom op  for [ModelOptmizer]() usage, you should consider `SOExtension`. More details about it can be found in [Library with Extensions](../../../../docs/Extensibility_UG/Intro.md#create-a-library-with-extensions).
### Python based extensions
C++ based extensions have their equivalents in Python. For `ConversionExtension` an example of usage can look like:
```
from openvino.frontend.onnx import ConversionExtension
...
def custom_add(node: NodeContext):
    input_1 = node.get_input(0)
    input_2 = node.get_input(1)
    add = ops.add(input_1, input_2)
    return [add.output(0)]

fe.add_extension(ConversionExtension("CustomAdd", "org.openvinotoolkit", custom_add))
```
Via `OpExtension` custom op registration can look like:
```
from openvino.frontend.onnx import OpExtension
...
fe.add_extension(OpExtension("opset9.Add", "CustomAdd", "org.openvinotoolkit", {}, {"auto_broadcast": "numpy"}))
```

## See also
 * [OpenVINO ONNX Frontend README](../README.md)
 * [OpenVINO™ README](../../../../README.md)
 * [Developer documentation](../../../../docs/dev/index.md)