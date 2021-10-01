// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_core.hpp>

int main() {
    //! [ie:create_core]
    InferenceEngine::Core core;
    //! [ie:create_core]

    //! [ie:read_model]
    InferenceEngine::CNNNetwork network = core.ReadNetwork("model.xml");
    //! [ie:read_model]

    //! [ie:get_inputs_outputs]
    InferenceEngine::InputsDataMap inputs = network.getInputsInfo();
    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    //! [ie:get_inputs_outputs]

    //! [ie:compile_model]
    InferenceEngine::ExecutableNetwork exec_network = core.LoadNetwork(network, "CPU");
    //! [ie:compile_model]

    //! [ie:create_infer_request]
    InferenceEngine::InferRequest infer_request = exec_network.CreateInferRequest();
    //! [ie:create_infer_request]

    //! [ie:get_input_tensor]
    InferenceEngine::Blob::Ptr input_blob = infer_request.GetBlob(inputs.begin()->first);
    // fill input blob
    //! [ie:get_input_tensor]

    //! [ie:inference]
    infer_request.Infer();
    //! [ie:inference]

    //! [ie:get_output_tensor]
    InferenceEngine::Blob::Ptr output_blob = infer_request.GetBlob(outputs.begin()->first);
    // process output data
    //! [ie:get_output_tensor]
    return 0;
}
