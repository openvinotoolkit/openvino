# OpenVINO™ Runtime API 2.0 Transition Guide {#openvino_2_0_transition_guide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:
   
   openvino_2_0_inference_pipeline
   openvino_2_0_configure_devices
   openvino_2_0_preprocessing
   openvino_2_0_model_creation
      
@endsphinxdirective

## Introduction

The OpenVINO™ 2.0 introduced to simplify migration of user applications from the frameworks like TensorFlow, PyTorch, ONNX, etc to OpenVINO runtime and make the OpenVINO™ API more user-friendly. This includes changes in several OpenVINO components:

- Previous versions of Model Optimizer were allowed to change the original input format of framework models:
  - Applying input precision changes for some types of models. For example, neural langauge processing models with `I64` input are becoming to have `I32` input element type.
  - Changing an order of dimensions (layout, see [Layouts in OpenVINO](../layout_overview.md)) for TensorFlow models. It leads to unexpected user behavior that a user needs to use a different layout for its input data with compare to the framework.
  - Model Optimizer required a user to specify input shapes for parameters for cases when framework's model has undefined shapes.
- Inference Engine API: InferenceEngine::CNNNetwork also applied some conversion rules for input and output element types which also quite unexpected by the users, so they have to configure additional preprocessing steps to align OpenVINO model to match the original framework's one.

OpenVINO Runtime API 2.0 is introduced and composed of Inference Engine API used for inference and ngraph API targeted to work with models, operations. The OpenVINO API 2.0 has common structure, naming convention styles, namespaces, removes duplicated structures. See [How to migrate to OpenVINO 2.0 API](./common_inference_pipeline.md) for details.

> **NOTE**: old ngraph and Inference Engine APIs are also preserved for backward compatibility and they are fully functional. The migration to OpenVINO 2.0 API is required to utilize the new OpenVINO Runtime API features like [Preprocessing](../preprocessing_overview.md) and [Dynamic shapes support](../DynamicBatching.md).

## Introduce IR v11

As a result of changes in Model Optimizer, OpenVINO introduced IR v11. From the user's perspective, the IR v11 looks like IR v10, but it has inputs and outputs formats aligned as it would be in the original framework. So, when a user converts a model, the converted model has exactly the same input element types, order of dimensions in shapes, also a user does not have to specify input shapes during the conversion, so the resulting IR v11 contains `-1` to denote undefined dimensions (see [Working with dynamic shapes](../DynamicBatching.md) to utilize this feature).

What is also important to mention - the IR v11 is fully compatible with old applications written using older versions of OpenVINO Runtime API - using Inference Engine API. This is achieved by adding additional runtime information to the IR v11 which is responsible for backwark compatible behavior. So, once the IR v11 is read by the old Inference Engine based application, it's internally converted to IR v10 to provide backward-compatible behavior.

## IR v10 compatibility

All user's applications written to work with IR v10 are also supported by OpenVINO Runtime API from OpenVINO 2.0. So, if a user has an IR v10, such IR v10 can be fed to OpenVINO Runtime as well (see [migration steps](./common_inference_pipeline.md)).

Some OpenVINO tools also support IR v10 as well as IR v11 as an input:
- [Compile tool](../../../tools/compile_tool/README.md) compiles the model to be used in OpenVINO 2.0 API by default. If a user wants to use the resulting compiled blob in Inference Engine API, the additional `ov_api_1_0` option should be passed.
- Accuracy checker also supports IR v10, but requires an additional option to denote which API is used underneath.

But the following OpenVINO model tools don't support IR v10 as an input, they require regenerated an IR v11 from the original model with latest Model Optimizer:
- Post Training Optimization tool
- Deep Learning WorkBench

## Differences between Inference Engine API and OpenVINO Runtime API 2.0

The list with differences between APIs below:

 - OpenVINO™ Runtime API 2.0 uses tensor names or indexes to work with Inputs or Outputs of the model, the old Inference Engine API works with operation names.
 - Structures for Shapes, element types were changed - structures from ngraph API was selected instead of Inference Engine API's ones.
 - Naming style convention was changed. The old Inference Engine API uses CamelCaseStyle and OpenVINO™ Runtime API 2.0 uses snake_case for function and variable names.
 - Namespaces were aligned between components and the new `ov` C++ namaspace is introduced instead of `ngraph` and `InferenceEngine`.

Let's define two types of behaviors:
- **Old behavior** of OpenVINO supposes:
  - Model Optimizer can change input element types, order of dimensions (layouts) with compare to the model from the original framework.
  - Inference Engine can override input and output element types.
  - Inference Engine API operates with operation names to address inputs and outputs (e.g. InferenceEngine::InferRequest::GetBlob).
  - Does not support compiling of models with dynamic input shapes.
- **New behavior** of OpenVINO runtime is implemented in OpenVINO 2.0:
  - Model Optimizer preserves the input element types, order of dimensions (layouts) and stores tensor names from the original models.
  - OpenVINO Runtime 2.0 reads models in any formats (IR v10, IR v11, ONNX, PaddlePaddle, etc) as is.
  - OpenVINO Runtime API 2.0 operates with tensor names. Note, the difference between tensor names and operations names is that in case if a single operation has several output tesnsors, such tensors cannot identified in a unique manner, so tensor names are used for addressing as it's usually done in the frameworks.
  - OpenVINO Runtime API 2.0 can address input and outputs tensors also by its index. Some model formats like ONNX are sensitive to order of inputs, outputs and its preserved by OpenVINO Runtime 2.0. 

The table below demonstrates which behavior **old** or **new** is used depending on a model source, used APIs.


| API  | IR v10  | IR v11  | ONNX file | Model created in code |
|------|-----------------------------------|
|Inference Engine / ngraph APIs | Old || Old | Old | Old |
|OpenVINO Runtime API 2.0 | Old | New | New | New |

Please look at next transition guides to understand how migrate Inference Engine-based application to OpenVINO™ Runtime API 2.0:
 - [OpenVINO™ Common Inference pipeline](common_inference_pipeline.md)
 - [Preprocess your model](./preprocessing.md)
 - [Configure device](./configure_devices.md)
 - [OpenVINO™ Model Creation](graph_construction.md)
 - [OMZ, POT APIs migration ?? ]
