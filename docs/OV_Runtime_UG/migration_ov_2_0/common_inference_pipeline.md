# Inference Pipeline {#openvino_2_0_inference_pipeline}

To infer models with OpenVINO™ Runtime, you usually need to perform the following steps in the application pipeline:
1. [Create a Core object](@ref create_core).
   - 1.1. [(Optional) Load extensions](@ref load_extensions)
2. [Read a model from a drive](@ref read_model).
   - 2.1. [(Optional) Perform model preprocessing](@ref perform_preprocessing).
3. [Load the model to the device](@ref load_model_to_device).
4. [Create an inference request](@ref create_inference_request).
5. [Fill input tensors with data](@ref fill_tensor).
6. [Start inference](@ref start_inference).
7. [Process the inference results](@ref process_results).

Based on the steps, the following code demostrates how to change the application code to migrate to API 2.0.

@anchor create_core
## 1. Create a Core Object

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:create_core
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:create_core
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:create_core
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_core
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:create_core
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:create_core
@endsphinxtab

@endsphinxtabset

@anchor load_extensions
### 1.1 (Optional) Load Extensions

To load a model with custom operations, you need to add extensions for these operations. It is highly recommended to use [OpenVINO Extensibility API](@ref openvino_docs_Extensibility_UG_Intro) to write extensions. However, you can also load the old extensions to the new OpenVINO™ Runtime:

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:load_old_extension
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:load_old_extension
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:load_old_extension
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:load_old_extension
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:load_old_extension
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:load_old_extension
@endsphinxtab

@endsphinxtabset

@anchor read_model
## 2. Read a Model from a Drive

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:read_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:read_model
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:read_model
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:read_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:read_model
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:read_model
@endsphinxtab

@endsphinxtabset

Reading a model has the same structure as the example in the [model creation migration guide](@ref openvino_2_0_model_creation).

You can combine reading and compiling a model into a single call `ov::Core::compile_model(filename, devicename)`.

@anchor perform_preprocessing
### 2.1 (Optional) Perform Model Preprocessing

When the application input data does not perfectly match the model input format, preprocessing may be necessary. See [preprocessing in API 2.0](@ref openvino_2_0_preprocessing) for more details.

@anchor load_model_to_device
## 3. Load the Model to the Device

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:compile_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:compile_model
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:compile_model
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:compile_model
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:compile_model
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:compile_model
@endsphinxtab

@endsphinxtabset

If you need to configure devices with additional parameters for OpenVINO Runtime, refer to [Configuring Devices](@ref openvino_2_0_configure_devices).

@anchor create_inference_request
## 4. Create an Inference Request

**Inference Engine API**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:create_infer_request
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:create_infer_request
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:create_infer_request
@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:create_infer_request
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:create_infer_request
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:create_infer_request
@endsphinxtab

@endsphinxtabset

@anchor fill_tensor
## 5. Fill Input Tensors with Data

**Inference Engine API**

The Inference Engine API fills inputs with data of the `I32` precision (**not** aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_input_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_input_tensor
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_input_tensor
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

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_input_tensor
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

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_input_tensor
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

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_input_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

API 2.0 fills inputs with data of the `I64` precision (aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_input_tensor_v10
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_input_tensor_v10
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_input_tensor_v10
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

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_input_tensor_aligned
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

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_input_tensor_aligned
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

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_input_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

@anchor start_inference
## 6. Start Inference

**Inference Engine API**

@sphinxtabset

@sphinxtab{Sync}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:inference
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:inference
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:inference
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

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:start_async_and_wait
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

@sphinxtabset

@sphinxtab{Sync}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:inference
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:inference
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:inference
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

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:start_async_and_wait
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

@anchor process_results

## 7. Process the Inference Results

**Inference Engine API**

The Inference Engine API processes outputs as they are of the `I32` precision (**not** aligned with the original model):

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ie_common.cpp ie:get_output_tensor
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ie_common.py ie:get_output_tensor
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_output_tensor
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

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_output_tensor
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

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_output_tensor
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

@sphinxtab{C}
@snippet docs/snippets/ie_common.c ie:get_output_tensor
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset

**API 2.0**

API 2.0 processes outputs as they are of:

- the `I32` precision (**not** aligned with the original model) for OpenVINO IR v10 models, to match the [old behavior](@ref differences_api20_ie).
- the `I64` precision (aligned with the original model) for OpenVINO IR v11, ONNX, ov::Model, PaddlePaddle and TensorFlow models, to match the [new behavior](@ref differences_api20_ie).

@sphinxtabset

@sphinxtab{IR v10}

@sphinxtabset

@sphinxtab{C++}
@snippet docs/snippets/ov_common.cpp ov_api_2_0:get_output_tensor_v10
@endsphinxtab

@sphinxtab{Python}
@snippet docs/snippets/ov_common.py ov_api_2_0:get_output_tensor_v10
@endsphinxtab

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_output_tensor_v10
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

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_output_tensor_aligned
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

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_output_tensor_aligned
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

@sphinxtab{C}
@snippet docs/snippets/ov_common.c ov_api_2_0:get_output_tensor_aligned
@endsphinxtab

@endsphinxtabset

@endsphinxtab

@endsphinxtabset
