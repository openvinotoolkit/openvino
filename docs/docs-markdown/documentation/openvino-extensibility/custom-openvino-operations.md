---
sidebar_position: 1
sidebar_label: "Custom OpenVINO Operations"
description: "Explore OpenVINO™ Extension API which enables registering custom operations to support models with operations not supported by OpenVINO."
---

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Custom OpenVINO Operations

OpenVINO™ Extension API allows you to register custom operations to support models with operations which OpenVINO™ does not support out-of-the-box.  A custom operation might be implemented both in C++ and Python.\`\`\`

Also it is possible to create a shared library with custom operation implemented in C++ first and load it using `add_extension` API. Please refer to [Create library with extensions](index.md#create-a-library-with-extensions) for more details on library creation and usage. The remaining part of this document describes how to implement an operation class using both the C++ API and Python API.

## Operation Class

<Tabs>
  <TabItem value="py" label="Python">
  To add your custom operation, create a new class that extends `openvino.Op`, which is in turn derived from `openvino.Node`, the base class for all graph operations in OpenVINO™. To add `openvino.Op` you need to import it.

  ```python
  from openvino import Op
  ```

  Follow the steps below to add a simple custom operation:

  1. Define the `__init__` method to initialize the class with inputs and attributes.
  2. Override the shape inference method `validate_and_infer_types`. This method is called multiple times during graph manipulations to determine the shapes and element types of the operations outputs. To access the input shapes and input element types, use the `get_input_partial_shape()` and `get_input_element_type()` methods of `openvino.Node`. Set the inferred shape and element type of the output using `set_output_type`.
  3. Override the `visit_attributes` method, which enables serialization and deserialization of operation attributes. An `AttributeVisitor` is passed to the method, and the implementation is expected to walk over all the attributes in the op using the type-aware `on_attribute` helper. Helpers are already implemented for standard types like `int`, `float`, `bool`, `vector`, and for existing OpenVINO defined types.
  4. Override `evaluate` method with the code that will run when this operation is encountered in the model graph during the model inference. It works only for CPU device and enables OpenVINO runtime to run your arbitrary Python code as a part of model inference. If your operation contains `evaluate` method you also need to override the `has_evaluate` method which returns True, this method allows to get information about availability of `evaluate` method for the operation.
  5. Override the `clone_with_new_inputs`, which is an optional method that graph manipulation routines to create copies of this operation and connect it to different nodes during optimization.
  </TabItem>
  <TabItem value="cpp" label="C++">
  To add your custom operation, create a new class that extends `ov::Op`, which is in turn derived from `ov::Node`, the base class for all graph operations in OpenVINO™. To add `ov::Op`, include the next file:

  ```cpp
  #include <openvino/op/op.hpp>
  ```

  Follow the steps below to add a custom operation:

  1. Add the `OPENVINO_OP` macro. The type info of an operation consists of a string operation identifier and a string for operation version.
  2. Implement default constructor and constructors that optionally take the operation inputs and attributes as parameters.
  3. Override the shape inference method `validate_and_infer_types`. This method is called multiple times during graph manipulations to determine the shapes and element types of the operations outputs. To access the input shapes and input element types, use the `get_input_partial_shape()` and `get_input_element_type()` methods of `ov::Node`. Set the inferred shape and element type of the output using `set_output_type`.
  4. Override the `clone_with_new_inputs` method, which enables graph manipulation routines to create copies of this operation and connect it to different nodes during optimization.
  5. Override the `visit_attributes` method, which enables serialization and deserialization of operation attributes. An `AttributeVisitor` is passed to the method, and the implementation is expected to walk over all the attributes in the op using the type-aware `on_attribute` helper. Helpers are already implemented for standard C++ types like `int64_t`, `float`, `bool`, `vector`, and for existing OpenVINO defined types.
  6. Override `evaluate` method, which enables fallback of some devices to this implementation and the application of constant folding if there is a custom operation on the constant branch. If your operation contains `evaluate` method you also need to override the `has_evaluate` method, this method allows to get information about availability of `evaluate` method for the operation.
  </TabItem>
</Tabs>

Based on that, declaration of an operation class can look as follows:

### Operation Constructors

OpenVINO™ operation contains two constructors:

- Default constructor, which enables you to create an operation without attributes
- Constructor that creates and validates an operation with specified inputs and attributes

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      def __init__(self, inputs=None, **attrs):
          super().__init__(self, inputs)
          self._attrs = attrs
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  Identity::Identity(const ov::Output<ov::Node>& arg) : Op({arg}) {
      constructor_validate_and_infer_types();
  }
  ```
  </TabItem>
</Tabs>

### `validate_and_infer_types()`

`ov::Node::validate_and_infer_types` method validates operation attributes and calculates output shapes using attributes of the operation.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      def validate_and_infer_types(self):
          self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  void Identity::validate_and_infer_types() {
      // Operation doesn't change shapes end element type
      set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
  }
  ```
  </TabItem>
</Tabs>

### `clone_with_new_inputs()`

`ov::Node::clone_with_new_inputs` method creates a copy of the operation with new inputs.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      def clone_with_new_inputs(self, new_inputs):
          return Identity(new_inputs)
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  std::shared_ptr<ov::Node> Identity::clone_with_new_inputs(const ov::OutputVector& new_args) const {
      OPENVINO_ASSERT(new_args.size() == 1, "Incorrect number of new arguments");

      return std::make_shared<Identity>(new_args.at(0));
  }
  ```
  </TabItem>
</Tabs>

### `visit_attributes()`

`ov::Node::visit_attributes` method enables you to visit all operation attributes.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      def visit_attributes(self, visitor):
          visitor.on_attributes(self._attrs)
          return True
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  bool Identity::visit_attributes(ov::AttributeVisitor& visitor) {
      return true;
  }
  ```
  </TabItem>
</Tabs>

### `evaluate() and has_evaluate()`

`ov::Node::evaluate` method enables you to apply constant folding to an operation.

<Tabs>
  <TabItem value="py" label="Python">
  ```python
      def evaluate(self, outputs, inputs):
          outputs[0].shape = inputs[0].shape
          inputs[0].copy_to(outputs[0])
          return True

      def has_evaluate(self):
          return True
  ```
  </TabItem>
  <TabItem value="cpp" label="C++">
  ```cpp
  bool Identity::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
      const auto& in = inputs[0];
      auto& out = outputs[0];
      if (out.data() == in.data())  // Nothing to do
          return true;
      out.set_shape(in.get_shape());
      memcpy(out.data(), in.data(), in.get_byte_size());
      return true;
  }

  bool Identity::has_evaluate() const {
      return true;
  }
  ```
  </TabItem>
</Tabs>
