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
#include "subgraph_tests/concat_split_relu.hpp"

namespace LayerTestsDefinitions {
    std::string ConcatSplitRelu::getTestCaseName(const testing::TestParamInfo<ConcatSplitReluTuple> &obj) {
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(netPrecision, targetName, additional_config) = obj.param;
        std::ostringstream results;

        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void ConcatSplitRelu::SetUp() {
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {{1, 256}});

        auto mem_c = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape{1, 256}, 0);
        auto mem_r = std::make_shared<ngraph::opset3::ReadValue>(mem_c, "id");

        auto cnc = std::make_shared<ngraph::op::Concat>(ngraph::OutputVector{input[0], mem_r}, 1);

        auto split1 = ngraph::builder::makeSplit(cnc, ngPrc, 2, 1);
        auto split2 = ngraph::builder::makeSplit(cnc, ngPrc, 2, 1);
        auto split3 = ngraph::builder::makeSplit(cnc, ngPrc, 2, 1);
        auto split4 = ngraph::builder::makeSplit(cnc, ngPrc, 2, 1);

        auto reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<size_t>{1, 128, 2});
        auto permute_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<size_t>{0, 2, 1});

        auto reshape1 = std::make_shared<ngraph::opset1::Reshape>(split2->output(1), reshape_pattern, false);
        auto reshape2 = std::make_shared<ngraph::opset1::Reshape>(split3->output(1), reshape_pattern, false);
        auto reshape3 = std::make_shared<ngraph::opset1::Reshape>(split4->output(1), reshape_pattern, false);

        auto permute1 = std::make_shared<ngraph::opset1::Transpose>(reshape1, permute_pattern);
        auto permute2 = std::make_shared<ngraph::opset1::Transpose>(reshape2, permute_pattern);
        auto permute3 = std::make_shared<ngraph::opset1::Transpose>(reshape3, permute_pattern);

        auto reshape_pattern_2 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, std::vector<size_t>{1, 256});
        auto reshape_4 = std::make_shared<ngraph::opset1::Reshape>(permute1, reshape_pattern_2, false);
        auto reshape_5 = std::make_shared<ngraph::opset1::Reshape>(permute2, reshape_pattern_2, false);
        auto reshape_6 = std::make_shared<ngraph::opset1::Reshape>(permute3, reshape_pattern_2, false);

        auto relu1 = std::make_shared<ngraph::opset1::Relu>(reshape_4);
        auto relu2 = std::make_shared<ngraph::opset1::Relu>(reshape_5);
        auto relu3 = std::make_shared<ngraph::opset1::Relu>(reshape_6);

        auto mem_w = std::make_shared<ngraph::opset3::Assign>(split1->output(1), "id");

        ngraph::ResultVector result{std::make_shared<ngraph::opset1::Result>(relu1),
                                    std::make_shared<ngraph::opset1::Result>(relu2),
                                    std::make_shared<ngraph::opset1::Result>(relu3),
                };
        mem_w->add_control_dependency(mem_r);
        relu2->add_control_dependency(mem_w);

        function = std::make_shared<ngraph::Function>(result, input, "concat_split_relu");
    }

    TEST_P(ConcatSplitRelu, CompareWithRefs) {
        auto ie = InferenceEngine::Core();
        cnnNetwork = InferenceEngine::CNNNetwork{function};

        auto exe_net = ie.LoadNetwork(cnnNetwork, targetDevice);
        auto inf_reg = exe_net.CreateInferRequest();
        ASSERT_NO_THROW(inf_reg.Infer());
};
} // namespace LayerTestsDefinitions
