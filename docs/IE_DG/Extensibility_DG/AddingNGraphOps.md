# Add Custom nGraph Operations {#openvino_docs_IE_DG_Extensibility_DG_AddingNGraphOps}

Inference Engine Extension API allows to register operation sets (opsets) with custom nGraph operations, it allows to support Networks with unknown operations.

## Operation Class

To add your custom nGraph operation, create a new class that extends `ngraph::Op`, which is in turn derived from `ngraph::Node`, the base class for all graph operations in nGraph. Follow the steps below:

1. Define a `NodeTypeInfo` object that identifies the type of the operation to the graph users and helps with dynamic type resolution. The type info of an nGraph operation currently consists of a string identifier and a version number, but this may change in the future.

2. Implement constructors that can optionally take the operation inputs and attributes as parameters. 

3. Override the shape inference method `validate_and_infer_types`. This method is called multiple times during graph manipulations to determine the shapes and element types of the outputs of the operations. You can access the input shapes through the `get_input_partial_shape()` method and input element types through the `get_input_element_type()` method of `ngraph::Node`. Set the inferred shape and element type of the output using `set_output_type`.

4. Override the `clone_with_new_inputs` method, which allows graph manipulation routines to create copies of this operation and connect it to different nodes during optimization.

5. Override the `visit_attributes` method, which allows serialization and deserialization of attributes. An `AttributeVisitor` is passed to the method, and the implementation is expected to walk over all the attributes in the op using the type-aware `on_attribute` helper. Helpers are already implemented for standard C++ types like `int64_t`, `float`, `bool`, `vector` and for existing nGraph defined types.

6. Override `evaluate`, which is an optional method that enables the application of constant folding if there is a custom operation on the constant branch.

Based on that, declaration of a operation class can look as follows:

@snippet op.hpp op:header

### Class Fields

The provided implementation has several fields:

 * `add` of type `int64_t` is an attribute of custom operation
 * `type_info` of type `ngraph::NodeTypeInfo` defines the type and version of operation

### Operation Constructors

nGraph operation contains two constructors: a default constructor, which allows to create operation without attributes and a constructor that creates and validates operation with specified inputs and attributes.

@snippet op.cpp op:ctor

### `validate_and_infer_types()`

`ngraph::Node::validate_and_infer_types` method validates operation attributes and calculates output shapes using attributes of operation.

@snippet op.cpp op:validate

### `clone_with_new_inputs()`

`ngraph::Node::clone_with_new_inputs` method creates a copy of nGraph operation with new inputs.

@snippet op.cpp op:copy

### `visit_attributes()`

`ngraph::Node::visit_attributes` method allows to visit all operation attributes.

@snippet op.cpp op:visit_attributes

### `evaluate()`

`ngraph::Node::evaluate` method allows to apply constant folding to an operation.

@snippet op.cpp op:evaluate

## Register Custom Operations in Extension Class

To add custom operations to the [Extension](Extension.md) class, create an operation set with custom operations and implement the `InferenceEngine::IExtension::getOpSets` method:

@snippet extension.cpp extension:getOpSets

This method returns a map of opsets that exist in the extension library.

nGraph provides opsets mechanism for operation versioning. Different opsets distinguish between different versions of one operation.

When specifying opset names, follow the rules below:
* Use unique opset names.
* Do not use the following built-in opset names: `extension`, `experimental`, `opset1`, `opest2`.
* Make sure that the Model Optimizer and your extension use the same opset names.
* IR v10 layers have the mandatory `version` attribute  specifying the opset. 
* `opset1` is the name of default operations set.
Operations from the default opset cannot be redefined.

Use a custom opset to create a new operation or extend functionality of an existing operation from another opset.
