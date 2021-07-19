// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "shared_test_classes/subgraph/multiple_input_fq.hpp"

namespace SubgraphTestsDefinitions {

std::string MultipleInputTest::getTestCaseName(const testing::TestParamInfo<multipleInputParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, config) = obj.param;
    std::ostringstream result;
    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void MultipleInputTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    std::tie(targetDevice, netPrecision, inputSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto input = ngraph::builder::makeParams(ngPrc, {{1, inputSize}, {1, inputSize}, {1, inputSize}});
    auto fake1 = ngraph::builder::makeFakeQuantize(input[0], ngPrc, 255, { 1 }, { -0.5 }, { 0.5 }, { -0.5 }, { 0.5 });
    auto mul1 = ngraph::builder::makeEltwise(input[0], fake1, ngraph::helpers::EltwiseTypes::ADD);
    auto fake2 = ngraph::builder::makeFakeQuantize(input[1], ngPrc, 255, { 1 }, { -0.5 }, { 0.5 }, { -0.5 }, { 0.5 });
    auto mul2 = ngraph::builder::makeEltwise(input[1], fake2, ngraph::helpers::EltwiseTypes::ADD);
    auto mul3 = ngraph::builder::makeEltwise(mul1, mul2, ngraph::helpers::EltwiseTypes::ADD);
    auto fake3 = ngraph::builder::makeFakeQuantize(input[2], ngPrc, 255, { 1 }, { -0.5 }, { 0.5 }, { -0.5 }, { 0.5 });
    auto mul4 = ngraph::builder::makeEltwise(fake3, mul3, ngraph::helpers::EltwiseTypes::ADD);
    auto result = std::make_shared<ngraph::opset7::Result>(mul4);
    function = std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, input, "multiple_input");
}

}  // namespace SubgraphTestsDefinitions

