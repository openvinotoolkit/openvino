// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/delayed_copy_layer.hpp"

namespace LayerTestsDefinitions {
    std::string DelayedCopyTest::getTestCaseName(const testing::TestParamInfo<ConcatSplitReluTuple> &obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(netPrecision, targetName, additional_config) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void DelayedCopyTest::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {{1, 384}});

        auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{1, 128}, std::vector<float>{0});

        auto mem_r = std::make_shared<ngraph::opset3::ReadValue>(mem_c, "id");

        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{mem_r, input[0]}, 1);
        auto split = ngraph::builder::makeVariadicSplit(concat, {384, 128}, 1);
        auto mem_w = std::make_shared<ngraph::opset3::Assign>(split->output(1), "id");

        auto VariadicSplit = ngraph::builder::makeVariadicSplit(concat, {64, 448}, 1);
        auto relu2 = std::make_shared<ngraph::opset1::Sigmoid>(VariadicSplit->output(1));

        mem_w->add_control_dependency(mem_r);
        relu2->add_control_dependency(mem_w);

        function = std::make_shared<ngraph::Function>(relu2, input, "delayed_copy_layer_memory");
    }

    void DelayedCopyTest::switchToNgraphFriendlyModel() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> config;
        std::map<std::string, std::string> additional_config;
        std::tie(netPrecision, targetDevice, additional_config) = this->GetParam();
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {{1, 384}});

        auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{1, 128}, std::vector<float>{0});
        auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{mem_c, input[0]}, 1);
        auto split = ngraph::builder::makeVariadicSplit(concat, {384, 128}, 1);

        auto VariadicSplit = ngraph::builder::makeVariadicSplit(concat, {64, 448}, 1);
        auto relu2 = std::make_shared<ngraph::opset1::Sigmoid>(VariadicSplit->output(1));

        function = std::make_shared<ngraph::Function>(relu2, input, "delayed_copy_layer_nonmemory");
    }

        void DelayedCopyTest::Run() {
        SKIP_IF_CURRENT_TEST_IS_DISABLED()

        LoadNetwork();
        Infer();
        switchToNgraphFriendlyModel();
        Validate();
    }

    TEST_P(DelayedCopyTest, CompareWithRefs) {
        Run();
};
} // namespace LayerTestsDefinitions
