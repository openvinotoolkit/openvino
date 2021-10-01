// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <openvino/core/function.hpp>
#include <openvino/runtime/runtime.hpp>

int main() {
    //! [ov_api_2_0:create_core]
    ov::runtime::Core core;
    //! [ov_api_2_0:create_core]

    //! [ov_api_2_0:read_model]
    std::shared_ptr<ov::Function> network = core.read_model("model.xml");
    //! [ov_api_2_0:read_model]

    //! [ov_api_2_0:get_inputs_outputs]
    ov::ParameterVector inputs = network->get_parameters();
    ov::ResultVector outputs = network->get_results();
    //! [ov_api_2_0:get_inputs_outputs]

    //! [ov_api_2_0:compile_model]
    ov::runtime::ExecutableNetwork exec_network = core.compile_model(network, "CPU");
    //! [ov_api_2_0:compile_model]

    ov::runtime::InferRequest infer_request = exec_network.create_infer_request();
    //
    // InferenceEngine::Blob::Ptr input_blob = infer_request.GetBlob(inputs.begin()->first);
    // // fill input blob
    // infer_request.Infer();
    //
    // InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(outputs.begin()->first);
    // process output data
    return 0;
}
