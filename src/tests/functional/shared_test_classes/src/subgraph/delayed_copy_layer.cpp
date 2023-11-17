// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/delayed_copy_layer.hpp"

namespace SubgraphTestsDefinitions {
    void DelayedCopyTestBase::InitMemory() {
        auto states = inferRequest.QueryState();
        for (auto& state : states) {
            auto name = state.GetName();
            if (name.find("id") != std::string::npos) {
                auto blob = FuncTestUtils::createAndFillBlobWithFloatArray(state.GetState()->getTensorDesc(),
                                                                           memory_init.data(), memory_init.size());
                state.SetState(blob);
            } else {
                GTEST_FAIL() << "unknown memory state";
            }
        }
    }

    void DelayedCopyTestBase::LoadNetwork() {
        LayerTestsUtils::LayerTestsCommon::LoadNetwork();
        inferRequest = executableNetwork.CreateInferRequest();
    }

    void DelayedCopyTestBase::Infer() {
        ConfigureInferRequest();
        inferRequest.Infer();
    }


    void DelayedCopyTestBase::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();
        InitMemory();
        GenerateInputs();
        Infer();
        switchToNgraphFriendlyModel();
        Validate();
    }

    std::string DelayedCopyTestBase::getTestCaseName(const testing::TestParamInfo<DelayedCopyTuple> &obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        size_t memory_size;
        std::tie(netPrecision, targetName, additional_config, memory_size) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        results << "memorySize=" << memory_size;
        for (auto const& configItem : additional_config) {
            results << "_configItem=" << configItem.first << "_" << configItem.second;
        }
        return results.str();
    }

    void DelayedCopyTest::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        size_t memory_size;
        std::tie(netPrecision, targetDevice, additional_config, memory_size) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());

        ASSERT_EQ(memory_size % 2, 0);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 3 * memory_size})};

        memory_init = ov::test::utils::generate_float_numbers(memory_size, -0.2f, 0.2f);

        auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{1, memory_size}, memory_init);

        auto mem_r = std::make_shared<ngraph::opset3::ReadValue>(mem_c, "id");

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{mem_r, input[0]}, 1);
        auto split = ngraph::builder::makeVariadicSplit(concat, {3 * memory_size, memory_size}, 1);
        auto mem_w = std::make_shared<ngraph::opset3::Assign>(split->output(1), "id");

        auto VariadicSplit = ngraph::builder::makeVariadicSplit(concat, {memory_size / 2, 3 * memory_size + memory_size / 2}, 1);
        auto relu2 = std::make_shared<ngraph::opset1::Sigmoid>(VariadicSplit->output(1));

        mem_w->add_control_dependency(mem_r);
        relu2->add_control_dependency(mem_w);

        function = std::make_shared<ngraph::Function>(relu2, input, "delayed_copy_layer_memory");
    }

    void DelayedCopyTest::switchToNgraphFriendlyModel() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> config;
        std::map<std::string, std::string> additional_config;
        size_t memory_size;
        std::tie(netPrecision, targetDevice, additional_config, memory_size) = this->GetParam();

        ASSERT_EQ(memory_size % 2, 0);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, 3 * memory_size})};

        auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{1, memory_size}, memory_init);
        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{mem_c, input[0]}, 1);
        auto split = ngraph::builder::makeVariadicSplit(concat, {3 * memory_size, memory_size}, 1);

        auto VariadicSplit = ngraph::builder::makeVariadicSplit(concat, {memory_size / 2, 3 * memory_size + memory_size / 2}, 1);
        auto relu2 = std::make_shared<ngraph::opset1::Sigmoid>(VariadicSplit->output(1));

        function = std::make_shared<ngraph::Function>(relu2, input, "delayed_copy_layer_nonmemory");
    }

    void DelayedCopyAfterReshapeWithMultipleConnTest::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        size_t memory_size;
        std::tie(netPrecision, targetDevice, additional_config, memory_size) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());

        ASSERT_EQ(memory_size % 8, 0);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, memory_size / 2})};

        memory_init = ov::test::utils::generate_float_numbers(memory_size, -0.2f, 0.2f);

        auto mem_c = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{8, memory_size / 8}, memory_init);
        auto mem_r = std::make_shared<ngraph::opset3::ReadValue>(mem_c, "id");
        auto reshape_pattern1 = ngraph::builder::makeConstant(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, memory_size});
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(mem_r, reshape_pattern1, false);
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(reshape1, split_axis_op, 2);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{split->output(0), input[0]}, 1);
        auto reshape_pattern2 = ngraph::builder::makeConstant(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{8, memory_size / 8});
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(concat, reshape_pattern2, false);

        auto mem_w = std::make_shared<ngraph::opset3::Assign>(reshape2, "id");

        auto relu = std::make_shared<ngraph::opset1::Sigmoid>(reshape2);
        auto reshape_pattern3 = ngraph::builder::makeConstant(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, memory_size});
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(relu, reshape_pattern3, false);

        mem_w->add_control_dependency(mem_r);
        reshape3->add_control_dependency(mem_w);

        function = std::make_shared<ngraph::Function>(reshape3, input, "delayed_copy_layer_reshape_memory");
    }

    void DelayedCopyAfterReshapeWithMultipleConnTest::switchToNgraphFriendlyModel() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> config;
        std::map<std::string, std::string> additional_config;
        size_t memory_size;
        std::tie(netPrecision, targetDevice, additional_config, memory_size) = this->GetParam();

        ASSERT_EQ(memory_size % 8, 0);

        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        ov::ParameterVector input {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, memory_size / 2})};

        auto mem_c = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1, memory_size}, memory_init);
        auto reshape_pattern1 = ngraph::builder::makeConstant(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, memory_size});
        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(mem_c, reshape_pattern1, false);
        auto split_axis_op = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto split = std::make_shared<ov::op::v1::Split>(reshape1, split_axis_op, 2);

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{split->output(0), input[0]}, 1);
        auto reshape_pattern2 = ngraph::builder::makeConstant(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{8, memory_size / 8});
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(concat, reshape_pattern2, false);

        auto relu = std::make_shared<ngraph::opset1::Sigmoid>(reshape2);
        auto reshape_pattern3 = ngraph::builder::makeConstant(ngraph::element::i64, ngraph::Shape{2}, ngraph::Shape{1, memory_size});
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(relu, reshape_pattern3, false);

        function = std::make_shared<ngraph::Function>(reshape3, input, "delayed_copy_layer_reshape_nonmemory");
    }
} // namespace SubgraphTestsDefinitions
