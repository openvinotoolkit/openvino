// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/concat_quantization_during_memory_requantization.hpp"

namespace SubgraphTestsDefinitions {
    std::string ConcatQuantDuringMemoryRequantTest::getTestCaseName(const testing::TestParamInfo<ConcatQuantDuringMemoryRequantTuple>& obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        size_t inputSize;
        size_t hiddenSize;
        std::map<std::string, std::string> config;
        std::tie(netPrecision, targetName, inputSize, hiddenSize, config) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "IS=" << inputSize << "_";
        results << "HS=" << hiddenSize << "_";
        results << "targetDevice=" << targetName;
        for (auto const& configItem : config) {
           results << "_configItem=" << configItem.second;
        }
        return results.str();
    }

    void ConcatQuantDuringMemoryRequantTest::SetUp() {
        InferenceEngine::Precision netPrecision;
        size_t inputSize;
        size_t hiddenSize;
        std::map<std::string, std::string> config;
        std::tie(netPrecision, targetDevice, inputSize, hiddenSize, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        memory_1_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.0f);
        memory_2_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.0f);

        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize})};

        auto mem_1_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize }, memory_1_init);
        auto mem_1_read = std::make_shared<ngraph::opset3::ReadValue>(mem_1_const, "memory_1");

        auto concat_1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ mem_1_read, input[0] }, 1);
        // Revert concat names to set the needed order of scale factors calculation
        concat_1->set_friendly_name("concat2");
        auto split_1 = ngraph::builder::makeVariadicSplit(concat_1, { inputSize, hiddenSize }, 1);

        auto mul_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize },
                                                                ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.0f));
        auto mul = ngraph::builder::makeEltwise(split_1->output(1), mul_const, ngraph::helpers::EltwiseTypes::MULTIPLY);
        auto mem_1_write = std::make_shared<ngraph::opset3::Assign>(mul, "memory_1");

        auto mem_2_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize }, memory_2_init);
        auto mem_2_read = std::make_shared<ngraph::opset3::ReadValue>(mem_2_const, "memory_2");

        auto concat_2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ mem_2_read, mul }, 1);
        // Revert concat names to set the needed order of scale factors calculation
        concat_2->set_friendly_name("concat1");
        auto split_2 = ngraph::builder::makeSplit(concat_2, ngPrc, 2, 1);
        auto mem_2_write = std::make_shared<ngraph::opset3::Assign>(split_2->output(0), "memory_2");
        auto sigm = std::make_shared<ngraph::opset1::Sigmoid>(split_2->output(1));

        mem_1_write->add_control_dependency(mem_1_read);
        sigm->add_control_dependency(mem_1_write);
        mem_2_write->add_control_dependency(mem_2_read);
        sigm->add_control_dependency(mem_2_write);

        function = std::make_shared<ngraph::Function>(sigm, input, "concat_quant_during_memory_requant_memory");
    }

    void ConcatQuantDuringMemoryRequantTest::switchToNgraphFriendlyModel() {
        InferenceEngine::Precision netPrecision;
        size_t inputSize;
        size_t hiddenSize;
        std::map<std::string, std::string> config;
        std::tie(netPrecision, targetDevice, inputSize, hiddenSize, config) = this->GetParam();
        configuration.insert(config.begin(), config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

        memory_1_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.0f);
        memory_2_init = ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.0f);

        ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize})};

        auto mem_1_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize }, memory_1_init);
        auto concat_1 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ mem_1_const, input[0] }, 1);
        auto split_1 = ngraph::builder::makeVariadicSplit(concat_1, { inputSize, hiddenSize }, 1);

        auto mul_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize },
                                                                ov::test::utils::generate_float_numbers(hiddenSize, -0.2f, 0.0f));
        auto mul = ngraph::builder::makeEltwise(split_1->output(1), mul_const, ngraph::helpers::EltwiseTypes::MULTIPLY);

        auto mem_2_const = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{ 1, hiddenSize }, memory_2_init);
        auto concat_2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ mem_2_const, mul }, 1);
        auto split_2 = ngraph::builder::makeSplit(concat_2, ngPrc, 2, 1);
        auto sigm = std::make_shared<ngraph::opset1::Sigmoid>(split_2->output(1));

        function = std::make_shared<ngraph::Function>(sigm, input, "concat_quant_during_memory_requant_nomemory");
    }

    void ConcatQuantDuringMemoryRequantTest::LoadNetwork() {
        LayerTestsUtils::LayerTestsCommon::LoadNetwork();
        inferRequest = executableNetwork.CreateInferRequest();
    }

    void ConcatQuantDuringMemoryRequantTest::Infer() {
        ConfigureInferRequest();
        inferRequest.Infer();
    }

    void ConcatQuantDuringMemoryRequantTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();

        auto states = inferRequest.QueryState();
        for (auto& state : states) {
            auto name = state.GetName();
            if (name == "memory_1") {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetState()->getTensorDesc(),
                                                                           memory_1_init.data(), memory_1_init.size());
                state.SetState(blob);
            } else if (name == "memory_2") {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetState()->getTensorDesc(),
                                                                           memory_2_init.data(), memory_2_init.size());
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
