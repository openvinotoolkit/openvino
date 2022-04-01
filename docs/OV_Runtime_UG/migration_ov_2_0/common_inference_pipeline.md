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

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:create_core
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:create_core
@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_core
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:create_core
@endsphinxtab

@endsphinxtabset

### 1.1 (Optional) Load extensions

To load a model with custom operations, you need to add extensions for these operations. We highly recommend using [OpenVINO Extensibility API](../../Extensibility_UG/Intro.md) to write extensions, but if you already have old extensions you can also load them to the new OpenVINO™ Runtime:

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:load_old_extension
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:load_old_extension
@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:load_old_extension
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:load_old_extension
@endsphinxtab

@endsphinxtabset

## 2. Read a model from a drive

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:read_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:read_model
@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:read_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:read_model
@endsphinxtab

@endsphinxtabset

Read model has the same structure as in the example from [Model Creation](./graph_construction.md) migration guide.

Note, you can combine read and compile model stages into a single call `ov::Core::compile_model(filename, devicename)`.

### 2.1 (Optional) Perform model preprocessing

When application's input data doesn't perfectly match the model's input format, preprocessing steps may be necessary.
See a detailed guide on [how to migrate preprocessing in OpenVINO Runtime API 2.0](./preprocessing.md)

## 3. Load the Model to the Device

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:compile_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:compile_model
@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:compile_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:compile_model
@endsphinxtab

@endsphinxtabset

If you need to configure OpenVINO Runtime devices with additional configuration parameters, refer to the [Configure devices](./configure_devices.md) guide.

## 4. Create an Inference Request

Inference Engine API:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:create_infer_request
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:create_infer_request
@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_infer_request
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:create_infer_request
@endsphinxtab

@endsphinxtabset

## 5. Fill input tensors

The Inference Engine API fills inputs as `I32` precision (**not** aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Model created in code}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0 fills inputs as `I64` precision (aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_v10
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_v10
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Model created in code}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

## 6. Start Inference

Inference Engine API:

@sphinxtabset

@sphinxtab{Sync}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:inference
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:inference
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Async}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:start_async_and_wait
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:start_async_and_wait
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0:

@sphinxtabset

@sphinxtab{Sync}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:inference
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:inference
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Async}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:start_async_and_wait
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:start_async_and_wait
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

## 7. Process the Inference Results

The Inference Engine API processes outputs as `I32` precision (**not** aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Model created in code}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

OpenVINO™ Runtime API 2.0 processes outputs:
- For IR v10 as `I32` precision (**not** aligned with the original model) to match the **old** behavior.
- For IR v11, ONNX, ov::Model, Paddle as `I64` precision (aligned with the original model) to match the **new** behavior.

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_v10
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_v10
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{IR v11}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{ONNX}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@sphinxtab{Model created in code}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset
