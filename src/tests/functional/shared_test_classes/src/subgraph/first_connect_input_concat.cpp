// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/first_connect_input_concat.hpp"


namespace SubgraphTestsDefinitions {

std::string ConcatFirstInputTest::getTestCaseName(const testing::TestParamInfo<concatFirstInputParams>& obj) {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, netPrecision, targetDevice, additional_config) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : additional_config) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void ConcatFirstInputTest::SetUp() {
    std::vector<std::vector<size_t>> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShapes, netPrecision, targetDevice, configuration) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params;
    for (auto&& shape : inputShapes) {
        params.push_back(std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shape)));
    }
    auto const_second_param = ngraph::builder::makeConstant(ngPrc, {1, 8}, std::vector<float>{-1.0f});
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{params[0], const_second_param}, 1);
    auto relu = std::make_shared<ngraph::opset1::Relu>(concat);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(relu)};

    function = std::make_shared<ngraph::Function>(results, params, "ConcatMultiInput");
}
}  // namespace SubgraphTestsDefinitions
