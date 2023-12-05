// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/parameter_reshape_result.hpp"

namespace SubgraphTestsDefinitions {
std::string ParamReshapeResult::getTestCaseName(const testing::TestParamInfo<ParamReshapeResultTuple> &obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> config;
    std::tie(inputShape, netPrecision, targetName, config) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    for (auto const& configItem : config) {
        results << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return results.str();
}

void ParamReshapeResult::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShape, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto shape = inputShape;
    shape[shape.size() - 2] *= 2;
    shape[shape.size() - 1] /= 2;
    auto reshape_const = std::make_shared<ngraph::opset8::Constant>(ngraph::element::Type_t::i64,
        ngraph::Shape{shape.size()}, shape);
    auto reshape = std::make_shared<ngraph::opset8::Reshape>(params[0], reshape_const, false);

    function = std::make_shared<ngraph::Function>(reshape, params, "ParamReshapeResult");
}
} // namespace SubgraphTestsDefinitions
