---
sidebar_position: 1
sidebar_label: "Model Representation in OpenVINO™ Runtime"
description: "In OpenVINO™ Runtime a model is represented by special classes to work with model data types and shapes."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Model Representation in OpenVINO™ Runtime

The model is represented by the `ov::Model class`.
The `ov::Model` object stores shared pointers to the following operations:

- `ov::op::v0::Parameter` - inputs
- `ov::op::v0::Result` - outputs
- `ov::op::Sink` - sinks (have no consumers and are not included in the results vector)

Other operations hold each other via shared pointers, where a child operation points to its
parent via a hard link. If an operation has no consumers and is neither a `Result` nor a
`Sink` operation whose shared pointer counter is zero, the operation will be removed.
Each operation in `ov::Model` has the `std::shared_ptr<ov::Node>` type.

## Classes and methods use to work with models

### Data types

The `ov::element::Type` is used as data type representation.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
  ov_input.get_element_type()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  ov_input->get_element_type();
  ```
  </TabItem>
</Tabs>

### Shapes

The OpenVINO runtime provides two types of shape representation:

- `ov::Shape` (read more) - represents static (fully defined) shapes,
- `ov::PartialShape` (read more) - represents shapes that may be partially or fully
  dynamic (with a given or an undefined rank).

`ov::PartialShape` can be converted to `ov::Shape`, but only if all dimensions are static.
Do this with the `get_shape()` method.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      partial_shape = node.output(0).get_partial_shape() # get zero output partial shape
      if not partial_shape.is_dynamic: # or partial_shape.is_static
          static_shape = partial_shape.get_shape()
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
      ov::Shape static_shape;
      ov::PartialShape partial_shape = node->output(0).get_partial_shape(); // get zero output partial shape
      if (!partial_shape.is_dynamic() /* or partial_shape.is_static() */) {
          static_shape = partial_shape.get_shape();
      }
  ```
  </TabItem>
</Tabs>

Keep in mind that OpenVINO enables you to change the model input shape during the application
runtime. It also supports models accepting dynamic shapes. For more details, see
[Model Input/Output Handling](model-input-output/index.md).

### Operations and Operation Sets

Each abstract operation in a model is represented by the `ov::Op class`.
A collection of operations used to construct a model is represented by the `ov::OpSet`
class. It refers to an operation set defined by an individual **opsetX** namespace,
for example **opset15**. These opsets are added with new OpenVINO releases to change the
behavior of previously used operations and help you avoid changing your application when new
operations are introduced. For more information, check the documentation on
[OpenVINO model](../../documentation/openvino-ir-format/index.md) and
[opsets](../../documentation/openvino-ir-format/operation-sets.md), as well as
[custom operations](../../documentation/openvino-extensibility/index.md).

## Build a Model in OpenVINO™ Runtime

You can create a model from source by constructing it with operations from an available
**opsetX** operation set. Each operation set integrates a list of pre-compiled operations
that work for this purpose. In other words, **opsetX** defines a set of operations for
building a model. To create an `ov::Model` instance from `opset15` operations, include
the following libraries:

<Tabs>
  <TabItem value="py" label="Python">
  ```cpp
  import openvino as ov
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  #include <openvino/core/model.hpp>
  #include <openvino/opsets/opset8.hpp>
  ```
  </TabItem>
</Tabs>

Then, you can create a model, as shown in these two examples:

<Tabs>
  <TabItem value="simple-single-output" label="Simple (single output)">
  <Tabs>
    <TabItem value="py" label="Python">
    ```cpp
    def create_simple_model():
        # This example shows how to create ov::Function
        #
        # Parameter--->Multiply--->Add--->Result
        #    Constant---'          /
        #              Constant---'
        data = ops.parameter([3, 1, 2], ov.Type.f32)
        mul_constant = ops.constant([1.5], ov.Type.f32)
        mul = ops.multiply(data, mul_constant)
        add_constant = ops.constant([0.5], ov.Type.f32)
        add = ops.add(mul, add_constant)
        res = ops.result(add)
        return ov.Model([res], [data], "model")
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    std::shared_ptr<ov::Model> create_simple_model() {
        // This example shows how to create ov::Model
        //
        // Parameter--->Multiply--->Add--->Result
        //    Constant---'          /
        //              Constant---'

        // Create opset8::Parameter operation with static shape
        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{3, 1, 2});

        auto mul_constant = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {1.5});
        auto mul = std::make_shared<ov::opset8::Multiply>(data, mul_constant);

        auto add_constant = ov::opset8::Constant::create(ov::element::f32, ov::Shape{1}, {0.5});
        auto add = std::make_shared<ov::opset8::Add>(mul, add_constant);

        // Create opset8::Result operation
        auto res = std::make_shared<ov::opset8::Result>(mul);

        // Create OpenVINO function
        return std::make_shared<ov::Model>(ov::ResultVector{res}, ov::ParameterVector{data});
    }
    ```
    </TabItem>
  </Tabs>
  </TabItem>
  <TabItem value="more-advanced-multiple-outputs" label="More advanced (multiple outputs)">
  <Tabs>
    <TabItem value="py" label="Python">
    ```cpp
    def create_advanced_model():
        # Advanced example with multi output operation
        #
        # Parameter->Split---0-->Result
        #               | `--1-->Relu-->Result
        #               `----2-->Result
        data = ops.parameter(ov.Shape([1, 3, 64, 64]), ov.Type.f32)
        # Create Constant for axis value
        axis_const = ops.constant(1, dtype=ov.Type.i64)

        # Create opset12::Split operation that splits input to three slices across 1st dimension
        split = ops.split(data, axis_const, 3)

        # Create opset12::Relu operation that takes 1st Split output as input
        relu = ops.relu(split.output(1))

        # Results operations will be created automatically based on provided OutputVector
        return ov.Model([split.output(0), relu.output(0), split.output(2)], [data], "model")
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    std::shared_ptr<ov::Model> create_advanced_model() {
        // Advanced example with multi output operation
        //
        // Parameter->Split---0-->Result
        //               | `--1-->Relu-->Result
        //               `----2-->Result

        auto data = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::Shape{1, 3, 64, 64});

        // Create Constant for axis value
        auto axis_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{} /*scalar shape*/, {1});

        // Create opset8::Split operation that splits input to three slices across 1st dimension
        auto split = std::make_shared<ov::opset8::Split>(data, axis_const, 3);

        // Create opset8::Relu operation that takes 1st Split output as input
        auto relu = std::make_shared<ov::opset8::Relu>(split->output(1) /*specify explicit output*/);

        // Results operations will be created automatically based on provided OutputVector
        return std::make_shared<ov::Model>(ov::OutputVector{split->output(0), relu, split->output(2)},
                                           ov::ParameterVector{data});
    }
    ```
    </TabItem>
  </Tabs>
  </TabItem>
</Tabs>

## Debug a Model

Here is a list of features that can help you debug models in OpenVINO:

- `DENABLE_OPENVINO_DEBUG=ON` - used to rebuild the OpenVINO Runtime library, to receive
  additional messages about applied model modifications
- `ov::pass::VisualizeTree` - used to save `ov::Model` to xDot format. It can be
  parametrized via environment variables:
  
  - `OV_VISUALIZE_TREE_OUTPUT_SHAPES=1` - visualize shapes
  - `OV_VISUALIZE_TREE_OUTPUT_TYPES=1` - visualize types
  - `OV_VISUALIZE_TREE_MIN_MAX_DENORMAL=1` - pretty denormal values
  - `OV_VISUALIZE_TREE_RUNTIME_INFO=1` - print runtime information
  - `OV_VISUALIZE_TREE_IO=1` - print I/O ports
  - `OV_VISUALIZE_TREE_MEMBERS_NAME=1` - print member names
- `ov::serialize` - used to save a model to IR:
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    def serialize_example(m : ov.Model):
        ov.serialize(m, xml_path='model.xml', bin_path='model.bin')
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    void serialize_example(const std::shared_ptr<ov::Model>& model) {
        ov::serialize(model, "/path/to/file/model.xml", "/path/to/file/model.bin");
    }
    ```
    </TabItem>
  </Tabs>
  
  > **Note:**
  >
  > While `save_model` is the preferred method for saving a model to IR, it is
  > recommended to use `serialize` for debugging purposes. It requires less time and
  > computational resources, as it does not apply weight compression.
- **xDot format to image conversion** - used to visualize model graphs:
  
  <Tabs>
    <TabItem value="py" label="Python">
    ```python
    def visualize_example(m : ov.Model):
        # Need import:
        # * import openvino.passes as passes
        pass_manager = passes.Manager()
        pass_manager.register_pass(passes.VisualizeTree(file_name='image.svg'))
        pass_manager.run_passes(m)
    ```
    </TabItem>
    <TabItem value="cpp" label="C++">
    ```cpp
    void visualize_example(const std::shared_ptr<ov::Model>& m) {
        // Need include:
        // * openvino/pass/manager.hpp
        // * openvino/pass/visualize_tree.hpp
        ov::pass::Manager manager;

        // Serialize ov::Model to before.svg file before transformation
        manager.register_pass<ov::pass::VisualizeTree>("image.svg");

        manager.run_passes(m);
    }
    ```
    </TabItem>
  </Tabs>

## Additional Resources

- [Available Operation Sets](../../../documentation/openvino-ir-format/operation-sets/available-opsets).
- [OpenVINO™ Runtime Extensibility Developer Guide](../../../documentation/openvino-extensibility).
- [Transformations Developer Guide](../../../documentation/openvino-extensibility/transformation-api).
