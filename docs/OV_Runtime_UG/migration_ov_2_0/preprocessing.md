# Preprocessing {#openvino_2_0_preprocessing}

### Introduction

Inference Engine API has preprocessing capabilities which are built on top of model expressed as `ngraph::Function` (now, it's `ov::Model`). Before OpenVINO Runtime API 2.0 preprocessing operations were not a part of the main model graph, so they were stored and executed separately:
- Preprocessing stored in `InferenceEngine::CNNNetwork` was lost during saving back to IR file format.
- Preprocessing operations are executed on CPU processors for most of the inference plugins. So, instead of occupying of acceleators, CPU processor is also busy with inference tasks.

OpenVINO Runtime API 2.0 introduces [new way of adding preprocessing operations to the model](../preprocessing_overview.md) - each preprocessing or postprocessing operation is integrated directly to the model and compiled together with inference part of the model via a single `ov::Core::compile_model` call.
Having preprocessing operations as a part of official OpenVINO opset allows to read and serialize preprocessed model to the IR file format.

In OpenVINO Runtime 2.0 the following new operations are introduced to be a part of OpenVINO `opset8`:
- [NV12toRGB](../../ops/image/NV12toRGB_8.md)
- [NV12toBGR](../../ops/image/NV12toBGR_8.md)
- [I420toBGR](../../ops/image/I420toBGR_8.md)
- [I420toRGB](../../ops/image/I420toRGB_8.md)

All other preprocessing operations are expressed by means of existing operations and their combinations.

It's also important to mention that since OpenVINO 2.0, the Runtime API does not assume any default layouts like Inference Engine did, for example `{ 1, 224, 224, 3 }` is supposed to be `NCHW` layout while it's not true. So, some preprocessing capabilities in OpenVINO Runtime API 2.0 requires explicitly set layouts, see [Layout overview](../layout_overview.md) how to do it. For example, to perform image scaling by partial dimensions `H` and `W`, preprocessing needs to know what dimensions are `H` and `W`.

> **NOTE**: Use Model Optimizer preprocessing capabilities to insert and optimize preprocessing operations to the model. In this case you don't need to read model in runtime application and set preprocessing, you can use [model caching feature](../Model_caching_overview.md) to improve time to inference stage.

The steps below demonstrates how to migrate preprocessing scenarios from Inference Engine API to OpenVINO Runtime API 2.0.
The examples below supposes we need to preprocess a model input with tensor name `tensor_name`, in Inferenece Engine API which used operation names to address the data, it's called `operation_name`.

### Mean and scale values

Inference Engine API:

@snippet docs/snippets/ov_preprocessing_migration.cpp mean_scale

OpenVINO Runtime API 2.0:

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_mean_scale

### Precision and layout conversions

Inference Engine API:

@snippet docs/snippets/ov_preprocessing_migration.cpp conversions

OpenVINO Runtime API 2.0:

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_conversions

### Color space conversions

Inference Engine API:

@snippet docs/snippets/ov_preprocessing_migration.cpp color_space

OpenVINO Runtime API 2.0:

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_color_space

### Image scaling

Inference Engine API:

@snippet docs/snippets/ov_preprocessing_migration.cpp image_scale

OpenVINO Runtime API 2.0:

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_image_scale

**See also:**
- [Preprocessing details](../preprocessing_details.md)
- [NV12 classification sample](../../../samples/cpp/hello_nv12_input_classification/README.md)
