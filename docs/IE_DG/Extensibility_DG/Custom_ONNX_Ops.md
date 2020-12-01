# Custom ONNX operators {#openvino_docs_IE_DG_Extensibility_DG_Custom_ONNX_Ops}

ONNX importer provides mechanism to register custom ONNX operators based on predefined or user-defined nGraph operations.
The function responsible for registering a new operator is called `ngraph::onnx_import::register_operator` and is defined in `onnx_import/onnx_utils.hpp`.

## Registering custom ONNX operator based on predefined nGraph operations

The steps below explain how to register a custom ONNX operator, for example, CustomRelu, in a domain called com.example.
CustomRelu is defined as follows:
```
x >= 0 => f(x) = x * alpha
x < 0  => f(x) = x * beta
```
where alpha, beta are float constants.

1. Include headers:
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:headers

2. Register the CustomRelu operator in the ONNX importer:
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:register_operator
The `register_operator` function takes four arguments: op_type, opset version, domain, and a function object.
The function object is a user-defined function that takes `ngraph::onnx_import::Node` as an input and based on that, returns a graph with nGraph operations.
The `ngraph::onnx_import::Node` class represents a node in ONNX model. It provides functions to fetch input node(s) (`get_ng_inputs`), fetch attribute value (`get_attribute_value`) and many more (please refer to `onnx_import/core/node.hpp` for full class declaration).
New operator registration must happen before the ONNX model is read, for example, if an ONNX model uses the 'CustomRelu' operator, `register_operator("CustomRelu", ...)` must be called before InferenceEngine::Core::ReadNetwork.
Re-registering ONNX operators within the same process is supported. During registration of the existing operator, a warning is printed.

The example below demonstrates an examplary model that requires previously created 'CustomRelu' operator:
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:model


For a reference on how to create a graph with nGraph operations, visit [nGraph tutorial](../nGraphTutorial.md).
For a complete list of predefined nGraph operators, visit [available operations sets](../../ops/opset.md).

If operator is no longer needed, it can be unregistered by calling `unregister_operator`. The function takes three arguments `op_type`, `version`, and `domain`.
@snippet onnx_custom_op/onnx_custom_op.cpp onnx_custom_op:unregister_operator

## Registering custom ONNX operator based on custom nGraph operations

The same principles apply when registering custom ONNX operator based on custom nGraph operations.
This example shows how to register custom ONNX operator based on `Operation` presented in [this tutorial](AddingNGraphOps.md), which is used in [TemplateExtension](Extension.md).
@snippet template_extension/extension.cpp extension:ctor

Here, the `register_operator` function is called in Extension's constructor, which makes sure that it is called before InferenceEngine::Core::ReadNetwork (since InferenceEngine::Core::AddExtension must be called before a model with custom operator is read).

The example below demonstrates how to unregister operator from Extension's destructor:
@snippet template_extension/extension.cpp extension:dtor
Note that it is mandatory to unregister custom ONNX operator if it is defined in dynamic shared library.

## Requirements for building with CMake

Program that uses the `register_operator` functionality, requires (in addition to Inference Engine) `ngraph` and `onnx_importer` libraries.
The `onnx_importer` is a component of `ngraph` package , so `find_package(ngraph REQUIRED COMPONENTS onnx_importer)` is sufficient to find both.
The `ngraph` package exposes two variables (`${NGRAPH_LIBRARIES}` and `${ONNX_IMPORTER_LIBRARIES}`), which reference `ngraph` and `onnx_importer` libraries.
Those variables need to be passed to the `target_link_libraries` command in the CMakeLists.txt file.

See below CMakeLists.txt for reference:
@snippet onnx_custom_op/CMakeLists.txt cmake:onnx_custom_op
