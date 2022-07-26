# Preprocessing {#openvino_2_0_preprocessing}

### Introduction

The Inference Engine API contains preprocessing capabilities in the `InferenceEngine::CNNNetwork` class. Such preprocessing information is not part of the main inference graph executed by [OpenVINO devices](../supported_plugins/Device_Plugins.md), so it is stored and executed separately before the inference stage.
- Preprocessing operations are executed on the CPU for most OpenVINO inference plugins. So, instead of occupying accelerators, they make CPU busy with computational tasks.
- Preprocessing information stored in `InferenceEngine::CNNNetwork` is lost when saving back to the IR file format.

OpenVINO Runtime API 2.0 introduces a [new way of adding preprocessing operations to the model](../preprocessing_overview.md) - each preprocessing or postprocessing operation is integrated directly into the model and compiled together with the inference graph.
- Add preprocessing operations first using `ov::preprocess::PrePostProcessor`
- Then, compile the model on the target using `ov::Core::compile_model`

Having preprocessing operations as part of an OpenVINO opset makes it possible to read and serialize a preprocessed model as the IR file format.

It is also important to mention that the OpenVINO Runtime API 2.0 does not assume any default layouts, like Inference Engine did. For example, both `{ 1, 224, 224, 3 }` and `{ 1, 3, 224, 224 }` shapes are supposed to be in the `NCHW` layout, while only the latter one is. So, some preprocessing capabilities in the API require layouts to be set explicitly. To learn how to do it, refer to [Layout overview](../layout_overview.md). For example, to perform image scaling by partial dimensions `H` and `W`, preprocessing needs to know what dimensions `H` and `W` are.

> **NOTE**: Use Model Optimizer preprocessing capabilities to insert preprocessing operations in you model for optimization. This way the application does not need to read the model and set preprocessing repeatedly, you can use the [model caching feature](../Model_caching_overview.md) to improve the time-to-inference.

The steps below demonstrate how to migrate preprocessing scenarios from the Inference Engine API to the OpenVINO Runtime API 2.0.
The snippets assume we need to preprocess a model input with the tensor name of `tensor_name` in the Inferenece Engine API, using operation names to address the data, called `operation_name`.

#### Importing preprocessing in Python

In order to utilize preprocessing, the following imports must be added.

Inference Engine API:

@snippet docs/snippets/ov_preprocessing_migration.py imports

OpenVINO Runtime API 2.0:

@snippet docs/snippets/ov_preprocessing_migration.py ov_imports

There are two different namespaces: `runtime`, which contains OpenVINO Runtime API classes; and `preprocess`, which provides the Preprocessing API.


### Mean and scale values

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp mean_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py mean_scale

@endsphinxtab

@endsphinxtabset

OpenVINO Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_mean_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_mean_scale

@endsphinxtab

@endsphinxtabset

### Precision and layout conversions

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp conversions

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py conversions

@endsphinxtab

@endsphinxtabset

OpenVINO Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_conversions

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_conversions

@endsphinxtab

@endsphinxtabset

### Image scaling

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp image_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py image_scale

@endsphinxtab

@endsphinxtabset

OpenVINO Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_image_scale

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_image_scale

@endsphinxtab

@endsphinxtabset

### Color space conversions

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp color_space

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py color_space

@endsphinxtab

@endsphinxtabset

OpenVINO Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}

@snippet docs/snippets/ov_preprocessing_migration.cpp ov_color_space

@endsphinxtab

@sphinxtab{Python}

@snippet  docs/snippets/ov_preprocessing_migration.py ov_color_space

@endsphinxtab

@endsphinxtabset


**See also:**
- [Preprocessing details](../preprocessing_details.md)
- [NV12 classification sample](../../../samples/cpp/hello_nv12_input_classification/README.md)
