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
    ov::runtime::Tensor input_tensor = infer_request.get_input_tensor();
    auto data = input_tensor.data<ov::element_type_traits<ov::element::u8>::value_type>();
    // Fill data ...
    //! [ov_api_2_0:get_input_tensor]

    //! [ov_api_2_0:inference]
    infer_request.infer();
    //! [ov_api_2_0:inference]

    //! [ov_api_2_0:get_output_tensor]
    ov::runtime::Tensor output_tensor = infer_request.get_tensor(*outputs.begin());
    // process output data
    //! [ov_api_2_0:get_output_tensor]
    return 0;
}
