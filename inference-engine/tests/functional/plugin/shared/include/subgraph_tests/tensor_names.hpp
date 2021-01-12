// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "shared_test_classes/subgraph/tensor_names.hpp"
#include <unordered_set>

namespace SubgraphTestsDefinitions {

TEST_P(TensorNamesTest, CheckTensorNames) {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    ConfigureNetwork();
    executableNetwork = core->LoadNetwork(cnnNetwork, targetDevice, configuration);

    auto inputs = cnnNetwork.getInputsInfo();
    auto outputs = cnnNetwork.getOutputsInfo();
    std::unordered_set<std::string> inNames;
    for (const auto& in : inputs)
        inNames.emplace(in.first);
    std::unordered_set<std::string> outNames;
    for (const auto& out : outputs)
        outNames.emplace(out.first);

    for (const auto& param : function->get_parameters()) {
        ASSERT_TRUE(inNames.count(cnnNetwork.getOVNameForOperation(param->get_friendly_name())));
        for (const auto& name : param->output(0).get_names())
            ASSERT_TRUE(inNames.count(cnnNetwork.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        ASSERT_TRUE(outNames.count(cnnNetwork.getOVNameForOperation(result->get_friendly_name())));
        for (const auto& name : result->input_value(0).get_names())
            ASSERT_TRUE(outNames.count(cnnNetwork.getOVNameForTensor(name)));
    }

    inferRequest = executableNetwork.CreateInferRequest();

    for (const auto& param : function->get_parameters()) {
        ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForOperation(param->get_friendly_name())));
        for (const auto& name : param->output(0).get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForOperation(result->get_friendly_name())));
        for (const auto& name : result->output(0).get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(cnnNetwork.getOVNameForTensor(name)));
    }
}

TEST_P(TensorNamesTest, CheckTensorNamesAfterClone) {
    cnnNetwork = InferenceEngine::CNNNetwork{function};
    InferenceEngine::CNNNetwork clonedNet(static_cast<InferenceEngine::ICNNNetwork::Ptr>(cnnNetwork));
    ConfigureNetwork();
    executableNetwork = core->LoadNetwork(clonedNet, targetDevice, configuration);

    auto inputs = clonedNet.getInputsInfo();
    auto outputs = clonedNet.getOutputsInfo();
    std::unordered_set<std::string> inNames;
    for (const auto& in : inputs)
        inNames.emplace(in.first);
    std::unordered_set<std::string> outNames;
    for (const auto& out : outputs)
        outNames.emplace(out.first);

    for (const auto& param : function->get_parameters()) {
        ASSERT_TRUE(inNames.count(clonedNet.getOVNameForOperation(param->get_friendly_name())));
        for (const auto& name : param->output(0).get_names())
            ASSERT_TRUE(inNames.count(clonedNet.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        ASSERT_TRUE(outNames.count(clonedNet.getOVNameForOperation(result->get_friendly_name())));
        for (const auto& name : result->input_value(0).get_names())
            ASSERT_TRUE(outNames.count(clonedNet.getOVNameForTensor(name)));
    }

    inferRequest = executableNetwork.CreateInferRequest();

    for (const auto& param : function->get_parameters()) {
        ASSERT_NO_THROW(inferRequest.GetBlob(clonedNet.getOVNameForOperation(param->get_friendly_name())));
        for (const auto& name : param->output(0).get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(clonedNet.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        ASSERT_NO_THROW(inferRequest.GetBlob(clonedNet.getOVNameForOperation(result->get_friendly_name())));
        for (const auto& name : result->output(0).get_names())
            ASSERT_NO_THROW(inferRequest.GetBlob(clonedNet.getOVNameForTensor(name)));
    }
}

}  // namespace SubgraphTestsDefinitions

