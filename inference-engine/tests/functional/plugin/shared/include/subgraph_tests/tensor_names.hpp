// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/tensor_names.hpp"
#include <unordered_set>

namespace SubgraphTestsDefinitions {

TEST_P(TensorNamesTest, CheckTensorNames) {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();

    auto inputs = cnnNetwork.getInputsInfo();
    auto outputs = cnnNetwork.getOutputsInfo();
    std::unordered_set<std::string> inNames;
    for (const auto& in : inputs)
        inNames.emplace(in.first);
    std::unordered_set<std::string> outNames;
    for (const auto& out : outputs)
        outNames.emplace(out.first);

    for (const auto& param : function->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names())
            ASSERT_TRUE(inNames.count(cnnNetwork.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        for (const auto& name : result->input_value(0).get_tensor().get_names())
            ASSERT_TRUE(outNames.count(cnnNetwork.getOVNameForTensor(name)));
    }

    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    inferRequest = executableNetwork.CreateInferRequest();

    for (const auto& param : function->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        for (const auto& name : result->get_input_tensor(0).get_names()) {
            ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForTensor(name)));
        }
    }
}

TEST_P(TensorNamesTest, CheckTensorNamesAfterClone) {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    IE_SUPPRESS_DEPRECATED_START
    InferenceEngine::CNNNetwork clonedNet(static_cast<InferenceEngine::ICNNNetwork::Ptr>(cnnNetwork));
    IE_SUPPRESS_DEPRECATED_END
    ConfigureNetwork();

    auto inputs = clonedNet.getInputsInfo();
    auto outputs = clonedNet.getOutputsInfo();
    std::unordered_set<std::string> inNames;
    for (const auto& in : inputs)
        inNames.emplace(in.first);
    std::unordered_set<std::string> outNames;
    for (const auto& out : outputs)
        outNames.emplace(out.first);

    for (const auto& param : function->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names())
            ASSERT_TRUE(inNames.count(clonedNet.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        for (const auto& name : result->get_input_tensor(0).get_names()) {
            ASSERT_TRUE(outNames.count(clonedNet.getOVNameForTensor(name)));
        }
    }

    executableNetwork = core->LoadNetwork(clonedNet, targetDevice, configuration);
    inferRequest = executableNetwork.CreateInferRequest();

    for (const auto& param : function->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(clonedNet.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        for (const auto& name : result->input_value(0).get_tensor().get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(clonedNet.getOVNameForTensor(name)));
    }
}

TEST_P(TensorNamesTest, CheckAddOutput) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();

    auto inputs = cnnNetwork.getInputsInfo();
    auto outputs = cnnNetwork.getOutputsInfo();
    std::unordered_set<std::string> inNames;
    for (const auto& in : inputs)
        inNames.emplace(in.first);
    std::unordered_set<std::string> outNames;
    for (const auto& out : outputs)
        outNames.emplace(out.first);

    ASSERT_EQ(1, inputs.size());
    ASSERT_EQ(1, outputs.size());
    ASSERT_EQ(1, function->get_results().size());

    // Check that relu_prev doesn't exist in output and input maps
    for (const std::string& tensor_name : {"relu,prev_t", "identity_prev_t"}) {
        ASSERT_THROW(cnnNetwork.getOVNameForTensor(tensor_name), InferenceEngine::NotFound);
    }

    // Add relu_prev as output
    cnnNetwork.addOutput("relu_prev");

    inputs = cnnNetwork.getInputsInfo();
    outputs = cnnNetwork.getOutputsInfo();
    inNames.clear();
    for (const auto& in : inputs)
        inNames.emplace(in.first);
    outNames.clear();
    for (const auto& out : outputs)
        outNames.emplace(out.first);

    ASSERT_EQ(1, inputs.size());
    ASSERT_EQ(2, outputs.size());
    ASSERT_EQ(2, function->get_results().size());

    // Check that relu_prev exists in output map
    for (const std::string& tensor_name : {"relu,prev_t", "identity_prev_t"}) {
        ASSERT_FALSE(inNames.count(cnnNetwork.getOVNameForTensor(tensor_name)));
    }
    for (const std::string& tensor_name : {"relu,prev_t", "identity_prev_t"}) {
        ASSERT_TRUE(outNames.count(cnnNetwork.getOVNameForTensor(tensor_name)));
    }

    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);
    inferRequest = executableNetwork.CreateInferRequest();

    for (const auto& param : cnnNetwork.getFunction()->get_parameters()) {
        for (const auto& name : param->get_output_tensor(0).get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForTensor(name)));
    }

    for (const auto& result : cnnNetwork.getFunction()->get_results()) {
        for (const auto& name : result->get_input_tensor(0).get_names()) {
            ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForTensor(name)));
        }
    }
}

}  // namespace SubgraphTestsDefinitions

