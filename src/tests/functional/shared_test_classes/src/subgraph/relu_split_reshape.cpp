// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/relu_split_reshape.hpp"

namespace SubgraphTestsDefinitions {
std::string ReluSplitReshape::getTestCaseName(const testing::TestParamInfo<ReluSplitReshapeTuple> &obj) {
    std::vector<size_t> inputShape;
    size_t splitAxis, splitNum;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> config;
    std::tie(inputShape, splitAxis, splitNum, netPrecision, targetName, config) = obj.param;
    std::ostringstream results;

    results << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    results << "axis=" << splitAxis << "_";
    results << "num=" << splitNum << "_";
    results << "netPRC=" << netPrecision.name() << "_";
    results << "targetDevice=" << targetName << "_";
    for (auto const& configItem : config) {
        results << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return results.str();
}

void ReluSplitReshape::SetUp() {
    std::vector<size_t> inputShape;
    size_t splitAxis, splitNum;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShape, splitAxis, splitNum, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto relu = std::make_shared<ngraph::opset1::Relu>(params[0]);
    auto split = ngraph::builder::makeSplit(relu, ngPrc, splitNum, splitAxis);

    auto shape = split->get_output_shape(0);
    shape[shape.size() - 2] *= 2;
    shape[shape.size() - 1] /= 2;
    auto reshape_const = std::make_shared<ngraph::opset7::Constant>(ngraph::element::Type_t::i64,
        ngraph::Shape{shape.size()}, shape);
    auto reshape = std::make_shared<ngraph::opset1::Reshape>(split->output(0), reshape_const, false);

    function = std::make_shared<ngraph::Function>(reshape, params, "ReluSplitReshape");
}
} // namespace SubgraphTestsDefinitions
