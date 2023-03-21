// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

include <c_api/ie_c_api.h>

static void completion_callback(void *args) {
    // Operations after infer
}

int main() {
    //! [ie:create_core]
    ie_core_t *core = nullptr;
    ie_core_create("", &core);
    //! [ie:create_core]

    //! [ie:read_model]
    ie_network_t *network = nullptr;
    ie_core_read_network(core, "model.xml", nullptr, &network);
    //! [ie:read_model]

    //! [ie:compile_model]
    ie_executable_network_t *exe_network = nullptr;
    ie_core_load_network(core, network, "CPU", nullptr, &exe_network);
    //! [ie:compile_model]

    //! [ie:create_infer_request]
    ie_infer_request_t *infer_request = nullptr;
    ie_exec_network_create_infer_request(exe_network, &infer_request);
    //! [ie:create_infer_request]

    char *input_name = nullptr;
    ie_network_get_input_name(network, 0, &input_name);
    //! [ie:get_input_tensor]
    // fill first blob
    ie_blob_t *input_blob1 = nullptr;
    {
    ie_infer_request_get_blob(infer_request, input_name, &input_blob1);
    ie_blob_buffer_t buffer;
    ie_blob_get_buffer(input_blob1, &buffer);
    // Original I64 precision was converted to I32
    int32_t* blob_internal_buffer = (int32_t*)buffer.buffer;
    // Fill data ...
    }
    // fill second blob
    ie_blob_t *input_blob2 = nullptr;
    {
    ie_infer_request_get_blob(infer_request, "data2", &input_blob2);
    ie_blob_buffer_t buffer;
    ie_blob_get_buffer(input_blob2, &buffer);
    // Original I64 precision was converted to I32
    int32_t* blob_internal_buffer = (int32_t*)buffer.buffer;
    // Fill data ...
    }

    //! [ie:get_input_tensor]

    //! [ie:inference]
    ie_infer_request_infer(infer_request);
    //! [ie:inference]

    //! [ie:start_async_and_wait]
    // NOTE: For demonstration purposes we are trying to set callback
    ie_complete_call_back_t callback;
    callback.completeCallBackFunc = completion_callback;
    callback.args = infer_request;
    ie_infer_set_completion_callback(infer_request, &callback);
    // Start inference without blocking current thread
    ie_infer_request_infer_async(infer_request);
    // Wait for 10 milisecond
    IEStatusCode waitStatus = ie_infer_request_wait(infer_request, 10);
    // Wait for inference completion
    ie_infer_request_wait(infer_request, -1);
    //! [ie:start_async_and_wait]

    //! [ie:get_output_tensor]
    // get output blob by name
    ie_blob_t *output_blob = nullptr;
    ie_infer_request_get_blob(infer_request, "output_name", &output_blob);
    // get blob buffer
    ie_blob_buffer_t out_buffer;
    ie_blob_get_buffer(output_blob, &out_buffer);
    // get data
    float *data = (float *)(out_buffer.buffer);
    // process output data
    //! [ie:get_output_tensor]

    //! [ie:load_old_extension]
    ie_core_add_extension(core, "path_to_extension_library.so", "CPU");
    //! [ie:load_old_extension]
    ie_blob_free(&output_blob);
    ie_blob_free(&input_blob2);
    ie_blob_free(&input_blob1);
    ie_network_name_free(&input_name);
    ie_infer_request_free(&infer_request);
    ie_exec_network_free(&exe_network);
    ie_network_free(&network);
    ie_core_free(&core);
    return 0;
}
