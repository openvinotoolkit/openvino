// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/core/core.hpp>
#include <openvino/runtime/runtime.hpp>

int main() {
    //! [ov_api_2_0:create_core]
    ov::runtime::Core core;
    //! [ov_api_2_0:create_core]

    //! [ov_api_2_0:read_model]
    std::shared_ptr<ov::Function> network = core.read_model("model.xml");
    //! [ov_api_2_0:read_model]

    //! [ov_api_2_0:get_inputs_outputs]
    std::vector<ov::Output<ov::Node>> inputs = network->inputs();
    std::vector<ov::Output<ov::Node>> outputs = network->outputs();
    //! [ov_api_2_0:get_inputs_outputs]

    //! [ov_api_2_0:compile_model]
    ov::runtime::ExecutableNetwork exec_network = core.compile_model(network, "CPU");
    //! [ov_api_2_0:compile_model]

    //! [ov_api_2_0:create_infer_request]
    ov::runtime::InferRequest infer_request = exec_network.create_infer_request();
    //! [ov_api_2_0:create_infer_request]

    //! [ov_api_2_0:get_input_tensor]
    // Get input tensor by index
    ov::runtime::Tensor input_tensor1 = infer_request.get_input_tensor(0);
    // IR v11 works with original precisions
    auto data1 = input_tensor1.data<int64_t>();
    // IR v10 works with converted precisions (i64 -> i32)
    auto data1_v10 = input_tensor1.data<int32_t>();
    // Fill first data ...

    // Get input tensor by tensor name
    ov::runtime::Tensor input_tensor2 = infer_request.get_tensor("data2_t");
    // IR v11 works with original precisions
    auto data2 = input_tensor1.data<int64_t>();
    // IR v10 works with converted precisions (i64 -> i32)
    auto data2_v10 = input_tensor1.data<int32_t>();
    // Fill first data ...
    //! [ov_api_2_0:get_input_tensor]

    //! [ov_api_2_0:inference]
    infer_request.infer();
    //! [ov_api_2_0:inference]

    //! [ov_api_2_0:get_output_tensor]
    // model has only one output
    ov::runtime::Tensor output_tensor = infer_request.get_output_tensor();
    // Get output tensor by port
    output_tensor = infer_request.get_tensor(outputs.at(0));
    // IR v11 works with original precisions
    auto out_data = output_tensor.data<int64_t>();
    // IR v10 works with converted precisions (i64 -> i32)
    auto out_data_v10 = output_tensor.data<int32_t>();
    // process output data
    //! [ov_api_2_0:get_output_tensor]
    return 0;
}
