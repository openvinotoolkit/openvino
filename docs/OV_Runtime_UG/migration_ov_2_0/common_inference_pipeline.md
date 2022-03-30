# Inference Pipeline {#openvino_2_0_inference_pipeline}

Usually, to infer models with OpenVINO™ Runtime, you need to make the following steps in the application pipeline:
- 1. Create Core object
 - 1.1. (Optional) Load extensions
- 2. Read a model from a drive
 - 2.1. (Optional) Perform model preprocessing
- 3. Load the model to the device
- 4. Create an inference request
- 5. Fill input tensors with data
- 6. Start inference
- 7. Process the inference results

The following code shows how to change the application code in each step to migrate to OpenVINO™ Runtime 2.0.

## 1. Create Core

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:create_core

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_core

### 1.1 (Optional) Load extensions

To load a model with custom operations, you need to add extensions for these operations. We highly recommend using [OpenVINO Extensibility API](../../Extensibility_UG/Intro.md) to write extensions, but if you already have old extensions you can also load them to the new OpenVINO™ Runtime:

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:load_old_extension

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:load_old_extension

## 2. Read a model from a drive

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:read_model

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:read_model

Read model has the same structure as in the example from [Model Creation](./graph_construction.md) migration guide.

Note, you can combine read and compile model stages into a single call `ov::Core::compile_model(filename, devicename)`.

### 2.1 (Optional) Perform model preprocessing

When application's input data doesn't perfectly match the model's input format, preprocessing steps may be necessary.
See a detailed guide on [how to migrate preprocessing in OpenVINO Runtime API 2.0](./preprocessing.md)

## 3. Load the Model to the Device

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:compile_model

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:compile_model

If you need to configure OpenVINO Runtime devices with additional configuration parameters, refer to the [Configure devices](./configure_devices.md) guide.

## 4. Create an Inference Request

Inference Engine API:

@snippet docs/snippets/ie_common.cpp ie:create_infer_request

OpenVINO™ Runtime API 2.0:

@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_infer_request

## 5. Fill input tensors

The Inference Engine API fills inputs as `I32` precision (**not** aligned with the original model):

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]

.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]

.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_input_tensor]

@endsphinxdirective

OpenVINO™ Runtime API 2.0 fills inputs as `I64` precision (aligned with the original model):

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_v10]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_aligned]

.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_aligned]

.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_input_tensor_aligned]

@endsphinxdirective

## 6. Start Inference

Inference Engine API:

@sphinxdirective

.. tab:: Sync

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:inference]

.. tab:: Async

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:start_async_and_wait]

@endsphinxdirective

OpenVINO™ Runtime API 2.0:

@sphinxdirective

.. tab:: Sync

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:inference]

.. tab:: Async

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:start_async_and_wait]

@endsphinxdirective

## 7. Process the Inference Results

The Inference Engine API processes outputs as `I32` precision (**not** aligned with the original model):

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]

.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]

.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ie_common.cpp
       :language: cpp
       :fragment: [ie:get_output_tensor]

@endsphinxdirective

OpenVINO™ Runtime API 2.0 processes outputs:
- For IR v10 as `I32` precision (**not** aligned with the original model) to match the **old** behavior.
- For IR v11, ONNX, ov::Model, Paddle as `I64` precision (aligned with the original model) to match the **new** behavior.

@sphinxdirective

.. tab:: IR v10

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_v10]

.. tab:: IR v11

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_aligned]

.. tab:: ONNX

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_aligned]

.. tab:: Model created in code

    .. doxygensnippet:: docs/snippets/ov_common.cpp
       :language: cpp
       :fragment: [ov_api_2_0:get_output_tensor_aligned]

@endsphinxdirective
