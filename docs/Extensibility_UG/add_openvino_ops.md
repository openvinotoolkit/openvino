# Custom OpenVINO™ Operations {#openvino_docs_Extensibility_UG_add_openvino_ops}

OpenVINO™ Extension API allows user to register custom operations. This allows to support models with operations which OpenVINO™ does not support out-of-the-box.

## Operation Class

To add custom operation, it is required to create a new class that extends `ov::Op`, which is in turn derived from `ov::Node` (the base class for all graph operations in OpenVINO™). To add `ov::Op` include next file:

@snippet template_extension/new/identity.hpp op:common_include

Follow the steps below to add a custom operation:

1. Add the `OPENVINO_OP` macro, which defines a `NodeTypeInfo` object that identifies the type of the operation to the graph users, and helps with dynamic type resolution. The type info of an operation currently consists of a string operation identifier, and a string for operation version;

2. Implement default constructor and constructors that optionally take the operation inputs and attributes as parameters;

3. Override the shape inference method `validate_and_infer_types`. This method is called multiple times during graph manipulations to determine the shapes, and element types of the operations outputs. To access the input shapes, and input element types, use the `get_input_partial_shape()`, and `get_input_element_type()` methods of `ov::Node`. Set the inferred shape and element type of the output using `set_output_type`.

4. Override the `clone_with_new_inputs` method, which enables graph manipulation routines to create copies of this operation and connect it to different nodes during optimization.

5. Override the `visit_attributes` method, which enables serialization, and deserialization of operation attributes. An `AttributeVisitor` is passed to the method, and the implementation is expected to walk over all the attributes in the op using the type-aware `on_attribute` helper. Helpers are already implemented for standard C++ types like `int64_t`, `float`, `bool`, `vector`, and for existing OpenVINO defined types.

6. Override `evaluate`, which is an optional method that enables fallback of some devices to this implementation, and the application of constant folding in case there is a custom operation on the constant branch. If your operation contains `evaluate` method, it is also required to override the `has_evaluate` method. This method allows to get information about availability of `evaluate` method for the operation.

Based on that, declaration of an operation class can look as follows:


### Operation Constructors

OpenVINO™ operation contains two constructors: 
* Default constructor -- it enables creation of an operation without attributes;
* Constructor -- it creates and validates an operation with specified inputs and attributes.

@snippet template_extension/new/identity.cpp op:ctor

### `validate_and_infer_types()`

The `ov::Node::validate_and_infer_types` method validates operation attributes and calculates output shapes by using attributes of the operation.

@snippet template_extension/new/identity.cpp op:validate

### `clone_with_new_inputs()`

The `ov::Node::clone_with_new_inputs` method creates a copy of the operation with new inputs.

@snippet template_extension/new/identity.cpp op:copy

### `visit_attributes()`

The `ov::Node::visit_attributes` method enables to visit all operation attributes.

@snippet template_extension/new/identity.cpp op:visit_attributes

### evaluate() and has_evaluate()

Th `ov::Node::evaluate` method enables to apply constant folding to an operation.

@snippet template_extension/new/identity.cpp op:evaluate

