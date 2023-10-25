// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/negative_memory_layer_offset.hpp"

namespace SubgraphTestsDefinitions {
    std::string NegativeMemoryOffsetTest::getTestCaseName(const testing::TestParamInfo<NegativeMemoryLayerOffsetTuple>& obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        size_t inputSize;
        size_t hiddenSize;
        std::tie(netPrecision, targetName, inputSize, hiddenSize, std::ignore) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "IS=" << inputSize << "_";
        results << "HS=" << hiddenSize << "_";
        results << "targetDevice=" << targetName;
        return results.str();
    }

    void NegativeMemoryOffsetTest::SetUp() {
        InferenceEngine::Precision netPrecision;
        size_t inputSize;
        size_t hiddenSize;
        std::map<std::string, std::string> config;
        std::tie(netPrecision, targetDevice, inputSize, hiddenSize, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        const int seed = 0;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
        for (size_t i = 0; i < hiddenSize; ++i)
            memory_init.emplace_back(static_cast<float>(dist(gen)));

        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize})};
        auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize }, memory_init);
        auto mem_r = std::make_shared<ngraph::opset3::ReadValue>(mem_c, "memory");

        // Use memory layer as the second input of 'concat' to get negative offset
        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ input[0], mem_r }, 1);
        auto split = ngraph::builder::makeVariadicSplit(concat, { hiddenSize, inputSize }, 1);
        auto mem_w = std::make_shared<ngraph::opset3::Assign>(split->output(0), "memory");
        auto sigm = std::make_shared<ngraph::opset1::Sigmoid>(split->output(1));

        mem_w->add_control_dependency(mem_r);
        sigm->add_control_dependency(mem_w);

        function = std::make_shared<ngraph::Function>(sigm, input, "negative_memory_layer_offset_memory");
    }

    void NegativeMemoryOffsetTest::switchToNgraphFriendlyModel() {
        InferenceEngine::Precision netPrecision;
        size_t inputSize;
        size_t hiddenSize;
        std::tie(netPrecision, targetDevice, inputSize, hiddenSize, std::ignore) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize})};
        auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize }, memory_init);
        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ input[0], mem_c }, 1);
        auto split = ngraph::builder::makeVariadicSplit(concat, { hiddenSize, inputSize }, 1);
        auto sigm = std::make_shared<ngraph::opset1::Sigmoid>(split->output(1));

        function = std::make_shared<ngraph::Function>(sigm, input, "negative_memory_layer_offset_nonmemory");
    }

    void NegativeMemoryOffsetTest::LoadNetwork() {
        LayerTestsUtils::LayerTestsCommon::LoadNetwork();
        inferRequest = executableNetwork.CreateInferRequest();
    }

    void NegativeMemoryOffsetTest::Infer() {
        ConfigureInferRequest();
        inferRequest.Infer();
    }

    void NegativeMemoryOffsetTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();
        auto states = inferRequest.QueryState();
        for (auto& state : states) {
            auto name = state.GetName();
            if (name == "memory") {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetState()->getTensorDesc(),
                                                                           memory_init.data(), memory_init.size());
                state.SetState(blob);
            } else {
                GTEST_FAIL() << "unknown memory state";
            }
        }
        GenerateInputs();
        Infer();
        switchToNgraphFriendlyModel();
        Validate();
    }
} // namespace SubgraphTestsDefinitions
