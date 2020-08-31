// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <debug.h>
#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/precision_utils.hpp"
#include "functional_test_utils/skip_tests_config.hpp"
#include "subgraph_tests/cascade_concat.hpp"

namespace LayerTestsDefinitions {

std::string CascadeConcat::getTestCaseName(const testing::TestParamInfo<CascadeConcatTuple> &obj) {
    std::vector<std::vector<size_t>> input1, input2, input3;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    bool multioutput;
    std::map<std::string, std::string> additional_config;
    std::tie(input1, input2, input3, netPrecision, multioutput, targetName, additional_config) = obj.param;
    std::ostringstream results;

    results << "IS=" << CommonTestUtils::vec2str(input1[0]) << "_";
    results << CommonTestUtils::vec2str(input2[0]) << "_";
    results << CommonTestUtils::vec2str(input3[0]) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "Multioutput=" << multioutput << "_";
    results << "targetDevice=" << targetName << "_";
    return results.str();
}

void CascadeConcat::SetUp() {
    std::vector<std::vector<size_t>> input1, input2, input3;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    bool multioutput;
    std::tie(input1, input2, input3, netPrecision, multioutput, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto input = ngraph::builder::makeParams(ngPrc, {input1[0], input2[0], input2[0]});
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(input[0]);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(input[1]);
    auto relu3 = std::make_shared<ngraph::opset1::Relu>(input[2]);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0),
                                                                                      relu2->output(0)},
                                                                                1);
    auto reshape = ngraph::builder::makeSqueezeUnsqueeze(concat, ngPrc, {0}, ngraph::helpers::SqueezeOpType::UNSQUEEZE);
    auto reshape2 = ngraph::builder::makeSqueezeUnsqueeze(reshape, ngPrc, {0}, ngraph::helpers::SqueezeOpType::SQUEEZE);
    auto concat2 = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{reshape2->output(0),
                                                                                       relu3->output(0)},
                                                                                 1);
    ngraph::ResultVector results;
    if (multioutput) {
        auto const_mult = ngraph::builder::makeConstant(ngPrc, ngraph::Shape{1, input1[0][1]+input2[0][1]},
                                                  std::vector<float>{1.01f});
        auto mult = std::make_shared<ngraph::op::v0::Multiply>(concat, const_mult);
        results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(concat2),
                                       std::make_shared<ngraph::opset1::Result>(mult)};
    } else {
        results = ngraph::ResultVector{std::make_shared<ngraph::opset1::Result>(concat2)};
    }
    function = std::make_shared<ngraph::Function>(results, input, "concat_reshape_reshape_concat_mul");
}

TEST_P(CascadeConcat, CompareWithRefs) {
    Run();
}
} // namespace LayerTestsDefinitions
