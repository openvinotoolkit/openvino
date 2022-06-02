# Preprocessing {#openvino_2_0_preprocessing}

This guide introduces how preprocessing works in API 2.0 by a comparison with preprocessing in the previous Inference Engine API. It also demonstrates how to migrate preprocessing scenarios from Inference Engine to API 2.0 via code samples.

## How Preprocessing Works in API 2.0

Inference Engine API contains preprocessing capabilities in the `InferenceEngine::CNNNetwork` class. Such preprocessing information is not a part of the main inference graph executed by [OpenVINO devices](../supported_plugins/Device_Plugins.md). Therefore, it is stored and executed separately before the inference stage:
- Preprocessing operations are executed on the CPU for most OpenVINO inference plugins. Thus, instead of occupying accelerators, they keep the CPU busy with computational tasks.
- Preprocessing information stored in `InferenceEngine::CNNNetwork` is lost when saving back to the OpenVINO IR file format.

API 2.0 introduces a [new way of adding preprocessing operations to the model](../preprocessing_overview.md) - each preprocessing or post-processing operation is integrated directly into the model and compiled together with the inference graph:
- API 2.0 first adds preprocessing operations by using `ov::preprocess::PrePostProcessor`,
- and then compiles the model on the target by using `ov::Core::compile_model`.

Having preprocessing operations as a part of an OpenVINO opset makes it possible to read and serialize a preprocessed model as the OpenVINO™ IR file format.

More importantly, API 2.0 does not assume any default layouts as Inference Engine did. For example, both `{ 1, 224, 224, 3 }` and `{ 1, 3, 224, 224 }` shapes are supposed to be in the `NCHW` layout, while only the latter is. Therefore, some preprocessing capabilities in the API require layouts to be set explicitly. To learn how to do it, refer to the [Layout overview](../layout_overview.md). For example, to perform image scaling by partial dimensions `H` and `W`, preprocessing needs to know what dimensions `H` and `W` are.

> **NOTE**: Use Model Optimizer preprocessing capabilities to insert preprocessing operations in your model for optimization. Thus, the application does not need to read the model and set preprocessing repeatedly. You can use the [model caching feature](../Model_caching_overview.md) to improve the time-to-inference.

The following sections demonstrate how to migrate preprocessing scenarios from Inference Engine API to API 2.0.
The snippets assume that you need to preprocess a model input with the `tensor_name` in Inference Engine API, using `operation_name` to address the data.

## Preparation: Import Preprocessing in Python

In order to utilize preprocessing, the following imports must be added.

**Inference Engine API**

@snippet docs/snippets/ov_preprocessing_migration.py imports

**API 2.0**

@snippet docs/snippets/ov_preprocessing_migration.py ov_imports

There are two different namespaces: 
- `runtime`, which contains API 2.0 classes;
- and `preprocess`, which provides Preprocessing API.

## Using Mean and Scale Values

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp mean_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py mean_scale

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_mean_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_mean_scale

@endsphinxtab

@endsphinxtabset

## Converting Precision and Layout

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp conversions

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py conversions

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_conversions

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_conversions

@endsphinxtab

@endsphinxtabset

## Using Image Scaling

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp image_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py image_scale

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_image_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_image_scale

@endsphinxtab

@endsphinxtabset

### Converting Color Space

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp color_space

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py color_space

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_color_space

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_color_space

@endsphinxtab

@endsphinxtabset


## Additional Resources

- [Preprocessing details](../preprocessing_details.md)
- [NV12 classification sample](../../../samples/cpp/hello_nv12_input_classification/README.md)
