// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/pass/low_latency.hpp>
#include "shared_test_classes/subgraph/basic_lstm.hpp"

namespace SubgraphTestsDefinitions {
TEST_P(Basic_LSTM_S, CompareWithRefImpl) {
    Run();
};

TEST_P(Basic_LSTM_S, CompareWithRefImpl_LowLatencyTransformation) {
    InferenceEngine::TensorDesc state_description(InferenceEngine::Precision::FP32,
                                                  InferenceEngine::SizeVector({1, hidden_size}),
                                                  InferenceEngine::Layout::NC);
    // Reshape
    auto params = ngraph::builder::makeParams(function->get_parameters().at(0)->get_element_type(), { {1, third_dim} });
    function->replace_parameter(0, params[0]);

    // todo: it is better to modify the model -> use ShapeOf() and Gather()
    std::vector<uint64_t> outFormShapes1 = { 1, 1, third_dim };
    auto pattern1 = std::make_shared<ngraph::opset1::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{3}, outFormShapes1);
    auto param_target_inputs = function->get_parameters().at(0)->output(0).get_target_inputs();

    // replace hardcoded shape
    for (const auto& target : param_target_inputs.begin()->get_node()->input(1).get_source_output().get_target_inputs()) {
        target.replace_source_output(pattern1);
    }
    function->validate_nodes_and_infer_types();

    // Generate inputs
    GenerateInputs();

    // Calculate References for the network before transformation passes
    auto referenceOutputs = CalculateRefs();

    // Apply LowLatency and UnrollTensorIterator transformations
    ngraph::pass::Manager manager;
    manager.register_pass<ngraph::pass::LowLatency2>(); // LowLatency enables UnrollTI
    manager.run_passes(function);
    LoadNetwork();
    IE_SUPPRESS_DEPRECATED_START
    auto states = executableNetwork.QueryState();
    for (auto& state : states) {
        auto name = state.GetName();
        if (name.find("cell_state_1") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       cell_memory_init.data(), cell_memory_init.size());
            state.SetState(blob);
        } else if (name.find("hidden_state_1") != std::string::npos) {
            auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state_description,
                                                                       hidden_memory_init.data(), hidden_memory_init.size());
            state.SetState(blob);
        } else {
            GTEST_FAIL() << "unknown memory state";
        }
    }
    IE_SUPPRESS_DEPRECATED_END
    // Run and compare
    Infer();
    const auto& actualOutputs = GetOutputs();
    Compare(referenceOutputs, actualOutputs);
};
}  // namespace SubgraphTestsDefinitions
