# Preprocessing {#openvino_2_0_preprocessing}

### Introduction

Inference Engine API has preprocessing capabilities in `InferenceEngine::CNNNetwork` class. Such preprocessing information is not a part of the main inference graph executed by the [OpenVINO devices](../supported_plugins/Device_Plugins.md), so it is stored and executed separately before an inference stage:
- Preprocessing operations are executed on CPU processor for most of the OpenVINO inference plugins. So, instead of occupying of acceleators, CPU processor is also busy with computational tasks.
- Preprocessing information stored in `InferenceEngine::CNNNetwork` is lost during saving back to IR file format.

OpenVINO Runtime API 2.0 introduces [new way of adding preprocessing operations to the model](../preprocessing_overview.md) - each preprocessing or postprocessing operation is integrated directly to the model and compiled together with inference graph:
- Add preprocessing operations first using `ov::preprocess::PrePostProcessor`
- Compile model on the target then using `ov::Core::compile_model`

Having preprocessing operations as a part of OpenVINO opset allows to read and serialize preprocessed model as the IR file format.

It's also important to mention that since OpenVINO 2.0, the Runtime API does not assume any default layouts like Inference Engine did, for example both `{ 1, 224, 224, 3 }` and `{ 1, 3, 224, 224 }` shapes are supposed to have `NCHW` layout while only the last shape has `NCHW`. So, some preprocessing capabilities in OpenVINO Runtime API 2.0 requires explicitly set layouts, see [Layout overview](../layout_overview.md) how to do it. For example, to perform image scaling by partial dimensions `H` and `W`, preprocessing needs to know what dimensions are `H` and `W`.

> **NOTE**: Use Model Optimizer preprocessing capabilities to insert and optimize preprocessing operations to the model. In this case you don't need to read model in runtime application and set preprocessing, you can use [model caching feature](../Model_caching_overview.md) to improve time to inference stage.

The steps below demonstrates how to migrate preprocessing scenarios from Inference Engine API to OpenVINO Runtime API 2.0.
The snippets suppose we need to preprocess a model input with tensor name `tensor_name`, in Inferenece Engine API using operation names to address the data, it's called `operation_name`.

### Mean and scale values

Inference Engine API:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [mean_scale]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [mean_scale]

@endsphinxdirective

OpenVINO Runtime API 2.0:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [ov_mean_scale]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [ov_mean_scale]

@endsphinxdirective

### Precision and layout conversions

Inference Engine API:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [conversions]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [conversions]

@endsphinxdirective

OpenVINO Runtime API 2.0:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [ov_conversions]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [ov_conversions]

@endsphinxdirective

### Image scaling

Inference Engine API:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [image_scale]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [image_scale]

@endsphinxdirective

OpenVINO Runtime API 2.0:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [ov_image_scale]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [ov_image_scale]

@endsphinxdirective

### Color space conversions

Inference Engine API:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [color_space]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [color_space]

@endsphinxdirective

OpenVINO Runtime API 2.0:

@sphinxdirective

.. tab:: C++

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.cpp
         :language: cpp
         :fragment: [ov_color_space]

.. tab:: Python

      .. doxygensnippet:: docs/snippets/ov_preprocessing_migration.py
         :language: python
         :fragment: [ov_color_space]

@endsphinxdirective

**See also:**
- [Preprocessing details](../preprocessing_details.md)
- [NV12 classification sample](../../../samples/cpp/hello_nv12_input_classification/README.md)
