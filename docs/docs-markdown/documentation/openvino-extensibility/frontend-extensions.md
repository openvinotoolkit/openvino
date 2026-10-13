---
sidebar_position: 2
sidebar_label: "Frontend Extensions"
description: "Learn how to use frontend extension classes to facilitate the mapping of custom operations from the framework model representation to the OpenVINO representation."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Frontend Extensions

The goal of this chapter is to explain how to use Frontend extension classes to facilitate
mapping of custom operations from framework model representation to OpenVINO representation.
Refer to [Introduction to OpenVINO Extension](index.md) to
understand the entire flow.

This API is applicable to new frontends only, which exist for ONNX, TensorFlow Lite, PaddlePaddle, and TensorFlow.

:::note
This documentation is written based on the [Template extension](https://github.com/openvinotoolkit/openvino/tree/master/src/core/template_extension),
which demonstrates extension development details based on minimalistic `Identity`
operation that is a placeholder for your real custom operation. You can review the complete code,
which is fully compilable, to see how it works.
:::

:::note
You can find more examples of extensions in [openvino\_contrib repository](https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations).
:::

## Single Operation Mapping with OpExtension

This section covers the case when a single operation in framework representation is mapped to a single
operation in OpenVINO representation. This is called *one-to-one mapping*. There is `OpExtension`
class that works well if all the following conditions are satisfied:

1. Number of inputs to operation in the Framework representation is the same as in the OpenVINO representation.
2. Number of outputs is also the same in both representations.
3. Inputs can be indexed and are mapped in order correspondingly, e.g.
   input with index 0 in framework representation maps to input with index 0 in OpenVINO representation and so on.
4. The same for outputs.
5. Each attribute in OpenVINO operation can be initialized from one of the attributes of original operation or by
   some predefined constant value. Value of copied attributes cannot contain expressions, value is accepted as-is,
   so type of a value should be compatible.

:::note
`OpExtension` class is currently available for ONNX and TensorFlow frontends.
PaddlePaddle frontend has named inputs and outputs for operation (not indexed)
therefore OpExtension mapping is not applicable for this case.
:::

The following example maps ONNX operation with the type of [Identity](https://github.com/onnx/onnx/blob/main/docs/Operators.md#Identity)
to OpenVINO template extension `Identity` class.

```cpp
#include <openvino/frontend/extension.hpp>
```

```cpp
auto extension1 = ov::frontend::OpExtension<TemplateExtension::Identity>("Identity");

// or even simpler if original FW type and OV type of operations match, that is "Identity"
auto extension2 = ov::frontend::OpExtension<TemplateExtension::Identity>();
```

The mapping doesn’t involve any attributes, as operation Identity doesn’t have them.

Extension objects, like just constructed `extension` can be used to add to the
OpenVINO runtime just before loading a model that contains custom operations:

```cpp
ov::Core core;
// Add arbitrary number of extensions before calling read_model method
core.add_extension(ov::frontend::OpExtension<TemplateExtension::Identity>());
core.read_model("/path/to/model.onnx");
```

However, extensions can also be constructed in a separately compiled shared library, that is suitable for loading models with custom operations in a Python application or tools like `benchmark_app`.
For details on how to build and load such library, check the following [guide](index.md#create-a-library-with-extensions).

If operation have multiple inputs and/or outputs they will be mapped in order.
The type of elements in input/output tensors should match expected types in the surrounding operations.
For example, if a custom operation produces the `f32` data type, the operation that consumes this output
should also support `f32`. Otherwise, model conversion fails with an error, as no automatic type conversion is performed.

### Converting to Standard OpenVINO Operation

`OpExtension` class can be used when mapping to one of the operations from standard OpenVINO
operation set is what you need and there is no class like `TemplateExtension::Identity` implemented.

Here is an example of a custom framework operation 'MyRelu'. Assume it is mathematically equivalent
to standard `Relu` that exists in the OpenVINO operation set, but for some reason has the type name of 'MyRelu'.
In this case, you can directly say that 'MyRelu' -> `Relu` mapping should be used:

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  from openvino.frontend import OpExtension
  core.add_extension(OpExtension("Relu", "MyRelu"))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  core.add_extension(ov::frontend::OpExtension<>("Relu", "MyRelu"));
  ```
  </TabItem>
</Tabs>

In the resulting converted OpenVINO model, “MyRelu” operation will be replaced by the standard operation
`Relu` from the latest available OpenVINO operation set. Notice that when standard operation is used,
it can be specified using just a type string (“Relu”) instead of using a `ov::opset8::Relu` class name as a
template parameter for `OpExtension`. This method is available for operations from the standard operation set only.
For a user custom OpenVINO operation the corresponding class should be always specified as a template parameter
as it was demonstrated with `TemplateExtension::Identity`.

### Attribute Mapping

As described above, `OpExtension` is useful when attributes can be mapped one by one or initialized by a constant.
Attributes in OpenVINO operators are identified by their names, so for frameworks that also have named attributes (like TensorFlow, PaddlePaddle, ONNX),
you can specify name to name mapping. For frameworks where OpenVINO operator's attributes can be mapped to one of the framework
operator inputs (like PyTorch), there's a name to input index mapping.

#### Named attributes mapping

If the set of attributes in framework representation and OpenVINO representation completely match by their names and types,
no attribute mapping has to be specified in OpExtension constructor parameters. The attributes are discovered and mapped automatically
based on `visit_attributes` method that should be defined for any OpenVINO operation.

Imagine you have CustomOperation class implementation that has two attributes with names: `attr1` and `attr2`.

```cpp
class CustomOperation : public ov::op::Op {

    std::string attr1;
    int attr2;

public:

    OPENVINO_OP("CustomOperation");

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("attr1", attr1);
        visitor.on_attribute("attr2", attr2);
        return true;
    }

    // ... implement other required methods
```

And original model in framework representation also has operation with name `CustomOperation` with the same
`attr1` and `attr2` attributes. Then with the following code:

```cpp
core.add_extension(ov::frontend::OpExtension<CustomOperation>());
```

Both `attr1` and `attr2` are copied from framework representation to OpenVINO representation automatically.

If for some reason names of attributes are different but values still can be copied “as-is” you can pass attribute
names mapping in `OpExtension` constructor:

```cpp
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"}, {"attr2", "fw_attr2"} },
    {}
));
```

Where `fw_attr1` and `fw_attr2` are names for corresponding attributes in framework operation representation.

If copying of an attribute is not what you need, `OpExtension` also can set attribute to predefined constant value.
For the same `CustomOperation`, imagine you want to set `attr2` to value 5 instead of copying from `fw_attr2`,
to achieve that do the following:

```cpp
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"} },
    { {"attr2", 5} }
));
```

So the conclusion is that each attribute of target OpenVINO operation should be initialized either by

1. Setting automatically due to name matching
2. Mapped by attribute name
3. Set to a constant value

This is achieved by specifying maps as arguments for `OpExtension` constructor.

#### Attribute mapping with named inputs and outputs

Mappings in previous examples assume that inputs and outputs of an operator in framework model representation come
with a particular order so you can directly map framework operation input `0` to OpenVINO operation input `0` and so on.
That's not always the case, for frameworks like PaddlePaddle, operation inputs and outputs are identified by their names
and may be defined in any order. So to map it to OpenVINO operation inputs and outputs, you have to specify that order yourself.
This can be done by creating two vector of strings, one for input and one for output, where framework operation
input name at position `i` maps to OpenVINO operation input at position `i` (and similarly for outputs).

Let's see the following example. Like previously, we'd like to map `CustomOperation` in the original model,
to OpenVINO `CustomOperation` as is (so their name and attributes names match). This time, that framework operation
inputs and outputs are not strictly ordered and can be identified  by their names `A`, `B`, `C` for inputs
and `X`, `Y` for outputs. Those inputs and outputs can be mapped to OpenVINO operation, such that inputs
`A`, `B`, `C` map to OpenVINO `CustomOperation` first, second and third input and `X` and `Y`
outputs map to OpenVINO `CustomOperation` first and second output respectively.

Given that, such custom operation can be registered by the following:

```cpp
core.add_extension(ov::frontend::OpExtension<CustomOperation>({"A", "B", "C"}, {"X", "Y"}));
```

Second example shows how to map the operation with named inputs and outputs, but when names of attributes are different:

```cpp
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    {"A", "B", "C"},
    {"X", "Y"},
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"}, {"attr2", "fw_attr2"} },
    {}
));
```

and the last one shows how to map the operation with named inputs and outputs, but when (in order to correctly map framework
operation to OpenVINO operation) one of the attributes has to be set to predefined value:

```cpp
core.add_extension(ov::frontend::OpExtension<CustomOperation>(
    {"A", "B", "C"},
    {"X", "Y"},
    std::map<std::string, std::string>{ {"attr1", "fw_attr1"} },
    { {"attr2", 5} }
));
```

#### Mapping attributes from operation inputs

For models (like PyTorch models), where operations have attributes on the input list, you can specify name to input index mapping.
For example, imagine you have created a custom OpenVINO operation that implements a variant of ELU activation function
with two attributes `alpha` and `beta`:

$$
CustomElu=\left\lbrace
\begin{array}{ll}
beta * x & \textrm{if x > 0} \newline
alpha * (exp(x) - 1) & \textrm{otherwise}
\end{array}
\right.
$$

Below is a snippet of `CustomElu` class showing how to define its attributes:

```cpp
class CustomElu : public ov::op::Op {
private:
    float m_alpha;
    float m_beta;

public:
    OPENVINO_OP("CustomElu");

    CustomElu() = default;

    CustomElu(const ov::Output<ov::Node>& input, float alpha, float beta) : Op({input}), m_alpha(alpha), m_beta(beta) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("alpha", m_alpha);
        visitor.on_attribute("beta", m_beta);
        return true;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<CustomElu>(inputs[0], m_alpha, m_beta);
    }
};
```

Let's see an example of how you can map `CustomElu` to PyTorch [aten::elu](https://pytorch.org/docs/stable/generated/torch.nn.functional.elu.html)
(note that if `beta` is equal to `1`, `CustomElu` works the same as `aten::elu`).
`aten::elu` has `alpha` attribute second on the input list, but it doesn't have `beta`.
So in order to map it to `CustomElu` you can use the following:

```cpp
auto extension = std::make_shared<ov::frontend::OpExtension<CustomElu>>("aten::elu",
                                                                        std::map<std::string, size_t>{{"alpha", 1}},
                                                                        std::map<std::string, ov::Any>{{"beta", 1.0f}});
```

This will map `alpha` to the second input and map `beta` attribute to constant value `1.0f`.

Such created extension can be used, e.g. in dynamic library, please refer to [Create a library with extensions](index.md#create-a-library-with-extensions).

## Mapping custom operations to frontends with OPENVINO\_FRAMEWORK\_MAP macro

`OPENVINO_FRAMEWORK_MAP` is a macro that should be used inside OpenVINO operation's class definition and that lets you specify
the mapping between this operation to a frontend operation.

Let's consider the following example. Imagine you have an ONNX model with `CustomOp` operation (and this operation has `mode` attribute),
a TensorFlow model with `CustomOpV3` operation (this operation has `axis` attribute) and a PaddlePaddle model with `CustomOp` (with `mode` attribute)
that has input named "X" and output named "Out" and all of them can be implemented with a single OpenVINO operation `CustomOp` like follows:

```cpp
#include <openvino/frontend/extension/op.hpp>
#include <openvino/frontend/onnx/extension/op.hpp>
#include <openvino/frontend/tensorflow/extension/op.hpp>
#include <openvino/frontend/paddle/extension/op.hpp>
```

```cpp
class CustomOp : public ov::op::Op {
    std::string m_mode;
    int m_axis;

public:
    OPENVINO_OP("CustomOp");
    OPENVINO_FRAMEWORK_MAP(onnx, "CustomOp", { {"mode", "mode"} }, { {"axis", -1} });
    OPENVINO_FRAMEWORK_MAP(tensorflow, "CustomOpV3", { {"axis", "axis"} }, { {"mode", "linear"} });
    OPENVINO_FRAMEWORK_MAP(paddle, {"X"}, {"Out"}, "CustomOp", { {"mode", "mode"} }, { {"axis", -1} });

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        visitor.on_attribute("mode", m_mode);
        visitor.on_attribute("axis", m_axis);
        return true;
    }

    // ... implement other required methods
```

Let's take a closer look at the parameters this macro takes (note that there are two flavors - the second one is to map
for PaddlePaddle operations where input and output names have to be specified).

```cpp
OPENVINO_FRAMEWORK_MAP(framework, name, attributes_map, attributes_values)
OPENVINO_FRAMEWORK_MAP(framework, input_names, output_names, name, attributes_map, attributes_values)
```

- `framework` - framework name.
- `name` - the framework operation name. It's optional if the OpenVINO custom operation name
  (that is the name that is passed as the first parameter to `OPENVINO_OP` macro) is the same
  as the framework operation name and both `attributes_map` and `attributes_values` are not provided.
- `input_names` - vector of strings that specify the names of inputs (needed to map PaddlePaddle to OpenVINO operations),
- `output_names` - vector of strings that specify the names of outputs (needed to map PaddlePaddle to OpenVINO operations),
- `attributes_map` - used to provide a mapping between OpenVINO operation attribute and
  framework operation attribute. Contains key-value pairs, where key is an OpenVINO operation
  attribute name and value is its corresponding framework operation attribute name.
  This parameter is optional if the number of OpenVINO operation attributes and their names
  match one-to-one with framework operation attributes.
- `attributes_values` - used to provide default values for OpenVINO operation attributes
  that are not specified in `attributes_map`. Contains key-value pairs, where key is an OpenVINO
  operation attribute name and the value is this attribute value. This parameter cannot be provided
  if `attributes_map` contains all of OpenVINO operation attributes or if `attributes_map` is not provided.

In the example above, `OPENVINO_FRAMEWORK_MAP` is used three times.
First, OpenVINO `CustomOp` is mapped to ONNX `CustomOp` operation, `m_mode` attribute is mapped to `mode`
attribute, while `m_axis` attribute gets the default value `-1`. Secondly, OpenVINO `CustomOp` is mapped
to TensorFlow `CustomOpV3` operation, `m_axis` attribute is mapped to `axis` attribute, while `m_mode`
attribute gets the default value `"linear"`. Thirdly, OpenVINO `CustomOp` is mapped to PaddlePaddle `CustomOp` operation,
`m_mode` attribute is mapped to `mode` attribute, while `m_axis` attribute gets the default value `-1`.
This mapping also specifies the input name "X" and output name "Out".

The last step is to register this custom operation by following:

```cpp
ov::Core core;
core.add_extension(ov::OpExtension<CustomOp>());
```

:::info[Important]
To map an operation on a specific framework, you have to link to a respective
frontend (`openvino::frontend::onnx`, `openvino::frontend::tensorflow`, `openvino::frontend::paddle`) in the `CMakeLists.txt` file:

```bash
target_link_libraries(${TARGET_NAME} PRIVATE openvino::frontend::onnx)
```
:::

## Mapping to Multiple Operations with ConversionExtension

Previous sections cover the case when a single operation is mapped to a single operation with optional
adjustment in names and attribute values. That is likely enough for your own custom operation with existing
C++ kernel implementation. In this case your framework representation and OpenVINO representation for the
operation are under your control and inputs/outpus/attributes can be aligned to make `OpExtension` usable.

In case if one-to-one mapping is not possible, *decomposition to multiple operations* should be considered.
It is achieved by using more verbose and less automated `ConversionExtension` class.
It enables writing arbitrary code to replace a single framework operation by multiple connected OpenVINO
operations constructing dependency graph of any complexity.

`ConversionExtension` maps a single operation to a function which builds a graph using OpenVINO
operation classes. Follow chapter \{ref\}`Build a Model in OpenVINO Runtime <ov_ug_build_model>` to
learn how to use OpenVINO operation classes to build a fragment of model for replacement.

Below example illustrates using `ConversionExtension` for conversion of “ThresholdedRelu”
from ONNX according to the formula: `ThresholdedRelu(x, alpha) -> Multiply(x, Convert(Greater(x, alpha), type=float))`.

:::note
`ThresholdedRelu` is one of the standard ONNX operators which is supported by ONNX frontend
natively out-of-the-box. Here we are re-implementing it to illustrate how you can add a similar
support for your custom operation instead of `ThresholdedRelu`.
:::

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  import openvino.opset14 as ops
  from openvino.frontend import ConversionExtension
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  #include <openvino/opsets/opset11.hpp>
  ```
  </TabItem>
</Tabs>

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  def conversion(node):
      input_node = node.get_input(0)
      input_type = input_node.get_element_type()
      greater = ops.greater(input_node, ops.constant([node.get_attribute("alpha")], input_type))
      casted = ops.convert(greater, input_type.get_type_name())
      return ops.multiply(input_node, casted).outputs()

  core.add_extension(ConversionExtension("ThresholdedRelu", conversion))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  core.add_extension(ov::frontend::ConversionExtension(
      "ThresholdedRelu",
      [](const ov::frontend::NodeContext& node) {
          auto greater = std::make_shared<ov::opset11::Greater>(
              node.get_input(0),
              ov::opset11::Constant::create(ov::element::f32, {}, {node.get_attribute<float>("alpha")}));
          auto casted = std::make_shared<ov::opset11::Convert>(greater, ov::element::f32);
          return ov::OutputVector{ std::make_shared<ov::opset11::Multiply>(node.get_input(0), casted) };
      }));
  ```
  </TabItem>
</Tabs>

The next example shows how to use `ConversionExtension` to convert PyTorch
[aten::hardtanh](https://pytorch.org/docs/stable/generated/torch.nn.functional.hardtanh.html)
to demonstrate how to use `get_values_from_const_input` function to fetch an attribute value from input:

```python
import torch
from openvino.frontend import ConversionExtension, NodeContext
from openvino import convert_model


class HardTanh(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(HardTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, inp):
        return torch.nn.functional.hardtanh(inp, self.min_val, self.max_val)


def convert_hardtanh(node: NodeContext):
    inp = node.get_input(0)
    min_value = node.get_values_from_const_input(1)
    max_value = node.get_values_from_const_input(2)
    return ops.clamp(inp, min_value, max_value).outputs()


model = HardTanh(min_val=0.1, max_val=2.0)
hardtanh_ext = ConversionExtension("aten::hardtanh", convert_hardtanh)
ov_model = convert_model(input_model=model, extension=[hardtanh_ext])
```

To access original framework operation attribute value and connect to inputs, `node` object of type `NodeContext` is used. It has three main methods:

- `NodeContext::get_input` to get input with a given index,
- `NodeContext::get_attribute` to get attribute value with a given name,
- `NodeContext::get_values_from_const_input` to get an attribute with a given input index.

The conversion function should return a vector of node outputs that are mapped to
corresponding outputs of the original framework operation in the same order.

Some frameworks require output names of the operation to be provided during conversion.
For PaddlePaddle operations, it is generally necessary to provide names for all outputs using the `NamedOutputs` container.
Usually those names can be found in source code of the individual operation in PaddlePaddle code.
The following example shows such conversion for the `top_k_v2` operation.

```cpp
core.add_extension(ov::frontend::ConversionExtension("top_k_v2", [](const ov::frontend::NodeContext& node) {
    auto x = node.get_input("X");
    const auto k_expected = node.get_attribute<int>("k", 1);
    auto k_expected_node = ov::opset11::Constant::create(ov::element::i32, {}, {k_expected});

    auto axis = node.get_attribute<int32_t>("axis", -1);
    bool sorted = node.get_attribute<bool>("sorted", true);
    bool largest = node.get_attribute<bool>("largest", true);

    std::string sort_type = sorted ? "value" : "none";
    std::string mode = largest ? "max" : "min";

    auto node_topk = std::make_shared<ov::opset11::TopK>(x, k_expected_node, axis, mode, sort_type);

    ov::frontend::paddle::NamedOutputs named_outputs;
    named_outputs["Out"] = ov::OutputVector{node_topk->output(0)};
    named_outputs["Indices"] = ov::OutputVector{node_topk->output(1)};

    return named_outputs;
}));
```

For TensorFlow framework, if an operation has more than one output, it is recommended to assign names to
those outputs using the `NamedOutputVector` structure which allows both indexed and named output access.
For a description of TensorFlow operations, including the names of their outputs, refer to the
[tf.raw\_ops](https://www.tensorflow.org/api_docs/python/tf/raw_ops/) documentation page.
The next example shows such conversion for the `TopKV2` operation.

```cpp
core.add_extension(ov::frontend::ConversionExtension("TopKV2", [](const ov::frontend::NodeContext& node) {
    auto input = node.get_input(0);
    auto k_input = node.get_input(1);
    bool sorted = node.get_attribute<bool>("sorted", true);
    auto mode = ov::opset11::TopK::Mode::MAX;
    auto sort_type = sorted ? ov::opset11::TopK::SortType::SORT_VALUES : ov::opset11::TopK::SortType::SORT_INDICES;
    auto top_k = std::make_shared<ov::opset11::TopK>(input, k_input, -1, mode, sort_type, ov::element::i32, true);
    return ov::frontend::NamedOutputVector{{"values", top_k->output(0)}, {"indices", top_k->output(1)}};
}));
```
