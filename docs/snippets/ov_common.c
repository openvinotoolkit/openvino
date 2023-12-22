// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/c/openvino.h>

void infer_request_callback(void* args) {
    // Operations after infer
}
void inputs_v10(ov_infer_request_t* infer_request) {
    //! [ov_api_2_0:get_input_tensor_v10]
    ov_tensor_t* input_tensor1 = NULL;
    ov_tensor_t* input_tensor2 = NULL;
    void* data = NULL;

    {
    // Get input tensor by index
    ov_infer_request_get_input_tensor_by_index(infer_request, 0, &input_tensor1);
    // IR v10 works with converted precisions (i64 -> i32)
    ov_tensor_data(input_tensor1, &data);
    int32_t* data1 = (int32_t*)data;
    // Fill first data ...
    }

    {
    // Get input tensor by tensor name
    ov_infer_request_get_tensor(infer_request, "data2_t", &input_tensor2);
    // IR v10 works with converted precisions (i64 -> i32)
    ov_tensor_data(input_tensor2, &data);
    int32_t* data2 = (int32_t*)data;
    // Fill first data ...
    }

    ov_tensor_free(input_tensor1);
    ov_tensor_free(input_tensor2);
    //! [ov_api_2_0:get_input_tensor_v10]
}

void inputs_aligned(ov_infer_request_t* infer_request) {
    //! [ov_api_2_0:get_input_tensor_aligned]
    ov_tensor_t* input_tensor1 = NULL;
    ov_tensor_t* input_tensor2 = NULL;
    void* data = NULL;
    {
    // Get input tensor by index
    ov_infer_request_get_input_tensor_by_index(infer_request, 0, &input_tensor1);
    // Element types, names and layouts are aligned with framework
    ov_tensor_data(input_tensor1, &data);
    // Fill first data ...
    }

    {
    // Get input tensor by tensor name
    ov_infer_request_get_tensor(infer_request, "data2_t", &input_tensor2);
    // Element types, names and layouts are aligned with framework
    ov_tensor_data(input_tensor2, &data);
    // Fill first data ...
    }

    ov_tensor_free(input_tensor1);
    ov_tensor_free(input_tensor2);
    //! [ov_api_2_0:get_input_tensor_aligned]
}

void outputs_v10(ov_infer_request_t* infer_request) {
    //! [ov_api_2_0:get_output_tensor_v10]
    ov_tensor_t* output_tensor = NULL;
    void* data = NULL;

    // model has only one output
    ov_infer_request_get_output_tensor(infer_request, &output_tensor);
    // IR v10 works with converted precisions (i64 -> i32)
    ov_tensor_data(output_tensor, &data);
    int32_t* out_data = (int32_t*)data;
    // process output data
    
    ov_tensor_free(output_tensor);
    //! [ov_api_2_0:get_output_tensor_v10]
}

void outputs_aligned(ov_infer_request_t* infer_request) {
    //! [ov_api_2_0:get_output_tensor_aligned]
    ov_tensor_t* output_tensor = NULL;
    void* out_data = NULL;
    
    // model has only one output
    ov_infer_request_get_output_tensor(infer_request, &output_tensor);
    // Element types, names and layouts are aligned with framework
    ov_tensor_data(output_tensor, &out_data);
    // process output data
    
    ov_tensor_free(output_tensor);
    //! [ov_api_2_0:get_output_tensor_aligned]
}

int main() {
    //! [ov_api_2_0:create_core]
    ov_core_t* core = NULL;
    ov_core_create(&core);
    //! [ov_api_2_0:create_core]

    //! [ov_api_2_0:read_model]
    ov_model_t* model = NULL;
    ov_core_read_model(core, "model.xml", NULL, &model);
    //! [ov_api_2_0:read_model]

    //! [ov_api_2_0:compile_model]
    ov_compiled_model_t* compiled_model = NULL;
    ov_core_compile_model(core, model, "CPU", 0, &compiled_model);
    //! [ov_api_2_0:compile_model]

    //! [ov_api_2_0:create_infer_request]
    ov_infer_request_t* infer_request = NULL;
    ov_compiled_model_create_infer_request(compiled_model, &infer_request);
    //! [ov_api_2_0:create_infer_request]

    inputs_aligned(infer_request);

    //! [ov_api_2_0:inference]
    ov_infer_request_infer(infer_request);
    //! [ov_api_2_0:inference]

    //! [ov_api_2_0:start_async_and_wait]
    // NOTE: For demonstration purposes we are trying to set callback
    ov_callback_t callback;
    callback.callback_func = infer_request_callback;
    callback.args = infer_request;
    ov_infer_request_set_callback(infer_request, &callback);
    // Start inference without blocking current thread
    ov_infer_request_start_async(infer_request);
    // Wait for inference completion
    ov_infer_request_wait(infer_request);
    // Wait for 10 milisecond
    ov_infer_request_wait_for(infer_request, 10);
    //! [ov_api_2_0:start_async_and_wait]

    outputs_aligned(infer_request);

    //! [ov_api_2_0:load_old_extension]
    // For C API 2.0 "add_extension()" is not supported for now
    //! [ov_api_2_0:load_old_extension]
    ov_infer_request_free(infer_request);
    ov_compiled_model_free(compiled_model);
    ov_model_free(model);
    ov_core_free(core);
    return 0;
}
