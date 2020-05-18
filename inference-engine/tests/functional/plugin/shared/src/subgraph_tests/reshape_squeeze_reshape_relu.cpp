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
#include "subgraph_tests/reshape_squeeze_reshape_relu.hpp"

namespace LayerTestsDefinitions {
    std::string ReshapeSqueezeReshapeRelu::getTestCaseName(const testing::TestParamInfo<ReshapeSqueezeReshapeReluTuple> &obj) {
        std::vector<std::vector<size_t>> input;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        bool is_squeeze;
        std::tie(input, netPrecision, targetName, is_squeeze) = obj.param;
        std::ostringstream results;

        results << "IS=" << CommonTestUtils::vec2str(input[0]) << "_";
        results << "indices=" << CommonTestUtils::vec2str(input[1]) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void ReshapeSqueezeReshapeRelu::SetUp() {
        std::vector<std::vector<size_t>> inputs;
        InferenceEngine::Precision netPrecision;
        bool is_squeeze;
        std::tie(inputs, netPrecision, targetDevice, is_squeeze) = this->GetParam();
        const std::size_t input_dim = InferenceEngine::details::product(inputs[0]);
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        std::vector<size_t> shape_input{1, input_dim};
        auto input = ngraph::builder::makeParams(ngPrc, {shape_input});
        auto reshape1_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{inputs[0].size()},
                                                                       inputs[0]);
        auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(input[0], reshape1_pattern, false);
        auto squeeze = [&]() {
            if (is_squeeze) {
                return ngraph::builder::makeSqueeze(reshape1, ngPrc, inputs[1]);
            }
            return ngraph::builder::makeUnsqueeze(reshape1, ngPrc, inputs[1]);
        };
        auto reshape2_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                                                                       ngraph::Shape{2},
                                                                       std::vector<size_t>{1, input_dim});
        auto reshape2 = std::make_shared<ngraph::op::v1::Reshape>(squeeze(), reshape2_pattern, false);
        auto func = std::make_shared<ngraph::opset1::Relu>(reshape2);
        if (is_squeeze)
            function = std::make_shared<ngraph::Function>(func, input, "reshape_squeeze_reshape_relu");
        else
            function = std::make_shared<ngraph::Function>(func, input, "reshape_unsqueeze_reshape_relu");
    }

    TEST_P(ReshapeSqueezeReshapeRelu, CompareWithRefs){
        Run();
    };
} // namespace LayerTestsDefinitions
