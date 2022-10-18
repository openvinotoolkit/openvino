// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

include <c_api/ie_c_api.h>

int main() {
    //! [ie:create_core]
    ie_core_t *core = nullptr;
    ie_core_create("", &core);
    //! [ie:create_core]

    //! [ie:read_model]
    ie_network_t *network = nullptr;
    ie_core_read_network(core, "model.xml", "model.bin", &network);
    //! [ie:read_model]

    //! [ie:compile_model]
    const char *device_name = "CPU";
    ie_config_t config = {nullptr, nullptr, nullptr};
    ie_executable_network_t *exe_network = nullptr;
    ie_core_load_network(core, network, device_name, &config, &exe_network);
    //! [ie:compile_model]

    //! [ie:create_infer_request]
    ie_infer_request_t *infer_request = nullptr;
    ie_exec_network_create_infer_request(exe_network, &infer_request);
    //! [ie:create_infer_request]

    //! [ie:get_input_tensor]
    char *input_name = nullptr;
    ie_network_get_input_name(network, 0, &input_name);
    ie_blob_t *blob = nullptr;
    ie_infer_request_get_blob(infer_request, input_name, &blob);
    {
    // fill first blob
    dimensions_t dims;
    IE_EXPECT_OK(ie_blob_get_dims(blob, &dims));
    const size_t blob_elems_count = dims.dims[0] * dims.dims[1] * dims.dims[2] * dims.dims[3];
    ie_blob_buffer_t buffer;
    IE_EXPECT_OK(ie_blob_get_buffer(blob, &buffer));
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
    ie_blob_t *output_blob = nullptr;
    ie_infer_request_get_blob(infer_request, "output_name", &output_blob);
    ie_blob_buffer_t out_buffer;
    ie_blob_get_buffer(output_blob, &out_buffer);
    void* data = nullptr;
    ov_tensor_data(output_blob, &data);
    //! [ie:get_output_tensor]

    //! [ie:load_old_extension]
    ie_core_add_extension(core, "path_to_extension_library.so", device_name);
    //! [ie:load_old_extension]
    return 0;
}
