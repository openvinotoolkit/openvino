# OpenVINO™ Inference Pipeline {#openvino_inference_pipeline}

Usually to inference network with the OpenVINO™ toolkit users need to do next steps:
 1. Create Core
 2. (Optional) Read model from the disk
     2.1. Configure Input and Output of the Model
 3. Load the Model to the Device
 4. Create an Inference Request
 5. Prepare Input
 6. Start Inference
 7. Process the Inference Results

Code snippets below cover these steps and show how application code should be changed for migration to OpenVINO™ 2.0.

## 1. Create Core

Inference Engine API:

@snippet snippets/ie_common.cpp ie:create_core

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:create_core

## 2. (Optional) Read model from the disk

Inference Engine API:

@snippet snippets/ie_common.cpp ie:read_model

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:read_model

Read model has the same structure as in the example from [OpenVINO™ Graph Construction](@ref openvino_graph_construction) guide.

### 2.1 Configure Input and Output of the Model

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_inputs_outputs

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_inputs_outputs

## 3. Load the Model to the Device

Inference Engine API:

@snippet snippets/ie_common.cpp ie:compile_model

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:compile_model

## 4. Create an Inference Request

Inference Engine API:

@snippet snippets/ie_common.cpp ie:create_infer_request

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:create_infer_request

## 5. Prepare input

### IR v10

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_input_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_input_tensor_v10

### IR v11

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_input_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned

### ONNX

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_input_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned

### From Function

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_input_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_input_tensor_aligned

## 6. Start Inference

Inference Engine API:

@snippet snippets/ie_common.cpp ie:inference

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:inference


## 7. Process the Inference Results

### IR v10

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_output_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_output_tensor_v10

### IR v11

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_output_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned

### ONNX

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_output_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned

### From Function

Inference Engine API:

@snippet snippets/ie_common.cpp ie:get_output_tensor

OpenVINO™ 2.0 API:

@snippet snippets/ov_common.cpp ov_api_2_0:get_output_tensor_aligned

