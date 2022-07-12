# Custom OpenVINO™ Operations {#openvino_docs_Extensibility_UG_add_openvino_ops}

OpenVINO™ Extension API allows you to register custom operations to support models with operations which OpenVINO™ does not support out-of-the-box.

> **NOTE**: Before you try adding a custom operation, make sure it hasn't already been added in a newer version of OpenVINO. A list of all operations supported in the latest version is available on the [Supported Framework Layers](../MO_DG/prepare_model/Supported_Frameworks_Layers.md) page. If it isn't supported by the latest version, there may also be pre-release versions of OpenVINO that have the operation (check the [nightly](https://docs.openvino.ai/nightly/openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers.html) page to see if the operation is supported in the nightly release). To upgrade to the latest version of OpenVINO, follow the [instructions here](../install_guides/installing-openvino-overview.md). To install pre-release versions of OpenVINO, see the the [GitHub Releases](https://github.com/openvinotoolkit/openvino/releases) page or the [Build From Source instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingCode).

## Operation Class

To add your custom operation, create a new class that extends `ov::Op`, which is in turn derived from `ov::Node`, the base class for all graph operations in OpenVINO™. To add `ov::Op` please include next file:

@snippet template_extension/new/identity.hpp op:common_include

Follow the steps below to add a custom operation:

1. Add the `OPENVINO_OP` macro which defines a `NodeTypeInfo` object that identifies the type of the operation to the graph users and helps with dynamic type resolution. The type info of an operation currently consists of a string operation identifier and a string for operation version.

2. Implement default constructor and constructors that optionally take the operation inputs and attributes as parameters. 

3. Override the shape inference method `validate_and_infer_types`. This method is called multiple times during graph manipulations to determine the shapes and element types of the operations outputs. To access the input shapes and input element types, use the `get_input_partial_shape()` and `get_input_element_type()` methods of `ov::Node`. Set the inferred shape and element type of the output using `set_output_type`.

4. Override the `clone_with_new_inputs` method, which enables graph manipulation routines to create copies of this operation and connect it to different nodes during optimization.

5. Override the `visit_attributes` method, which enables serialization and deserialization of operation attributes. An `AttributeVisitor` is passed to the method, and the implementation is expected to walk over all the attributes in the op using the type-aware `on_attribute` helper. Helpers are already implemented for standard C++ types like `int64_t`, `float`, `bool`, `vector`, and for existing OpenVINO defined types.

6. Override `evaluate`, which is an optional method that enables fallback of some devices to this implementation and the application of constant folding if there is a custom operation on the constant branch. If your operation contains `evaluate` method you also need to override the `has_evaluate` method, this method allows to get information about availability of `evaluate` method for the operation.

Based on that, declaration of an operation class can look as follows:


### Operation Constructors

OpenVINO™ operation contains two constructors: 
* Default constructor, which enables you to create an operation without attributes 
* Constructor that creates and validates an operation with specified inputs and attributes

@snippet template_extension/new/identity.cpp op:ctor

### `validate_and_infer_types()`

`ov::Node::validate_and_infer_types` method validates operation attributes and calculates output shapes using attributes of the operation.

@snippet template_extension/new/identity.cpp op:validate

### `clone_with_new_inputs()`

`ov::Node::clone_with_new_inputs` method creates a copy of the operation with new inputs.

@snippet template_extension/new/identity.cpp op:copy

### `visit_attributes()`

`ov::Node::visit_attributes` method enables you to visit all operation attributes.

@snippet template_extension/new/identity.cpp op:visit_attributes

### evaluate() and has_evaluate()

`ov::Node::evaluate` method enables you to apply constant folding to an operation.

@snippet template_extension/new/identity.cpp op:evaluate

