// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/core/core.hpp>
#include <openvino/runtime/runtime.hpp>

void inputs_v10(ov::InferRequest& infer_request) {
    //! [ov_api_2_0:get_input_tensor_v10]
    // Get input tensor by index
    ov::Tensor input_tensor1 = infer_request.get_input_tensor(0);
    // IR v10 works with converted precisions (i64 -> i32)
    auto data1 = input_tensor1.data<int32_t>();
    // Fill first data ...

    // Get input tensor by tensor name
    ov::Tensor input_tensor2 = infer_request.get_tensor("data2_t");
    // IR v10 works with converted precisions (i64 -> i32)
    auto data2 = input_tensor1.data<int32_t>();
    // Fill first data ...
    //! [ov_api_2_0:get_input_tensor_v10]
}

void inputs_aligned(ov::InferRequest& infer_request) {
    //! [ov_api_2_0:get_input_tensor_aligned]
    // Get input tensor by index
    ov::Tensor input_tensor1 = infer_request.get_input_tensor(0);
    // Element types, names and layouts are aligned with framework
    auto data1 = input_tensor1.data<int64_t>();
    // Fill first data ...

    // Get input tensor by tensor name
    ov::Tensor input_tensor2 = infer_request.get_tensor("data2_t");
    // Element types, names and layouts are aligned with framework
    auto data2 = input_tensor1.data<int64_t>();
    // Fill first data ...
    //! [ov_api_2_0:get_input_tensor_aligned]
}

void outputs_v10(ov::InferRequest& infer_request) {
    //! [ov_api_2_0:get_output_tensor_v10]
    // model has only one output
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    // IR v10 works with converted precisions (i64 -> i32)
    auto out_data = output_tensor.data<int32_t>();
    // process output data
    //! [ov_api_2_0:get_output_tensor_v10]
}

void outputs_aligned(ov::InferRequest& infer_request) {
    //! [ov_api_2_0:get_output_tensor_aligned]
    // model has only one output
    ov::Tensor output_tensor = infer_request.get_output_tensor();
    // Element types, names and layouts are aligned with framework
    auto out_data = output_tensor.data<int64_t>();
    // process output data
    //! [ov_api_2_0:get_output_tensor_aligned]
}

int main() {
    //! [ov_api_2_0:create_core]
    ov::Core core;
    //! [ov_api_2_0:create_core]

    //! [ov_api_2_0:read_model]
    std::shared_ptr<ov::Model> model = core.read_model("model.xml");
    //! [ov_api_2_0:read_model]

    //! [ov_api_2_0:compile_model]
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    //! [ov_api_2_0:compile_model]

    //! [ov_api_2_0:create_infer_request]
    ov::InferRequest infer_request = compiled_model.create_infer_request();
    //! [ov_api_2_0:create_infer_request]

    inputs_aligned(infer_request);

    //! [ov_api_2_0:set_callback]
    auto restart_once = true;
    infer_request.set_callback([&, restart_once] (std::exception_ptr exception_ptr) mutable {
        if (exception_ptr) {
            // procces exception or rethrow it.
            std::rethrow_exception(exception_ptr);
        } else {
            // Extract inference result
            ov::Tensor output_tensor = infer_request.get_output_tensor();
            // Restart inference if needed
            if (restart_once) {
                infer_request.start_async();
                restart_once = false;
            }
        }
    });
    //! [ov_api_2_0:set_callback]

    //! [ov_api_2_0:inference]
    infer_request.infer();
    //! [ov_api_2_0:inference]

    //! [ov_api_2_0:start_async_and_wait]
    // Start inference without blocking current thread
    infer_request.start_async();
    // Get inference status
    bool status = infer_request.wait_for(std::chrono::milliseconds{0});
    // Wait for one miliseconds
    status = infer_request.wait_for(std::chrono::milliseconds{1});
    // Wait for inference complition
    infer_request.wait();
    //! [ov_api_2_0:start_async_and_wait]

    outputs_aligned(infer_request);

    return 0;
}