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
        for (const auto& name : param->get_output_tensor(0).get_names())
            ASSERT_TRUE(inNames.count(cnnNetwork.getOVNameForTensor(name)));
    }

    for (const auto& result : function->get_results()) {
        ASSERT_TRUE(outNames.count(cnnNetwork.getOVNameForOperation(result->get_friendly_name())));
        for (const auto& name : result->get_input_tensor(0).get_names())
            ASSERT_TRUE(outNames.count(cnnNetwork.getOVNameForTensor(name)));
    }
}

}  // namespace SubgraphTestsDefinitions

