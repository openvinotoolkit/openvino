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
#include "subgraph_tests/multioutput_eltwise_squeeze_eltwise.hpp"

namespace LayerTestsDefinitions {
    std::string MultioutputEltwiseReshapeEltwise::getTestCaseName(const testing::TestParamInfo<MultioutputEltwiseReshapeEltwiseTuple> &obj) {
        std::vector<std::vector<size_t>> input;
        InferenceEngine::Precision netPrecision;
        std::string targetName;
        std::map<std::string, std::string> additional_config;
        std::tie(input, netPrecision, targetName, additional_config) = obj.param;
        std::ostringstream results;

        results << "IS=" << CommonTestUtils::vec2str(input[0]) << "_";
        results << "netPRC=" << netPrecision.name() << "_";
        results << "targetDevice=" << targetName << "_";
        return results.str();
    }

    void MultioutputEltwiseReshapeEltwise::SetUp() {
        std::vector<std::vector<size_t>> inputs;
        InferenceEngine::Precision netPrecision;
        std::map<std::string, std::string> additional_config;
        std::tie(inputs, netPrecision, targetDevice, additional_config) = this->GetParam();
        configuration.insert(additional_config.begin(), additional_config.end());
        auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
        auto input = ngraph::builder::makeParams(ngPrc, {inputs});

        auto eltwise_const = ngraph::builder::makeConstant(ngPrc,
                                                    ngraph::Shape{input[0]->get_shape()},
                                                     std::vector<float>{-1.0f});
        auto eltwise = std::make_shared<ngraph::opset1::Multiply>(input[0], eltwise_const);
        auto squeeze = ngraph::builder::makeUnsqueeze(eltwise, ngPrc, {0});
        auto unsqueeze = ngraph::builder::makeSqueeze(squeeze, ngPrc, {0});
        auto eltwise_const2 = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1}, std::vector<float>{1.01f});
        auto eltwise_const3 = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1}, std::vector<float>{1.01f});
        auto eltwise2 = std::make_shared<ngraph::opset1::Multiply>(eltwise, eltwise_const2);
        auto eltwise3 = std::make_shared<ngraph::opset1::Multiply>(unsqueeze, eltwise_const3);
        ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(eltwise2),
                                     std::make_shared<ngraph::opset1::Result>(eltwise3)};
        function = std::make_shared<ngraph::Function>(results, input, "eltwise_reshape_eltwise_multioutput");
    }

    TEST_P(MultioutputEltwiseReshapeEltwise, CompareWithRefs){
        Run();
    };
} // namespace LayerTestsDefinitions
