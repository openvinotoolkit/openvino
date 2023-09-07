// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/trivial_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string TrivialConcatLayerTest::getTestCaseName(const testing::TestParamInfo<trivialConcatParamsTuple> &obj) {
    std::vector<size_t> inputShapes;
    InferenceEngine::Precision netPrecision;
    std::string targetName;
    std::map<std::string, std::string> config;
    std::tie(inputShapes, netPrecision, targetName, config) = obj.param;
    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "trgDev=" << targetName << "_";
    return result.str();
}

void TrivialConcatLayerTest::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> additional_config;
    std::tie(inputShape, netPrecision, targetDevice, additional_config) = this->GetParam();
    configuration.insert(additional_config.begin(), additional_config.end());
    int axis = inputShape.size() - 2;
    size_t total_size = std::accumulate(inputShape.begin(), inputShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, total_size})};

    auto input_relu = ngraph::builder::makeActivation(params[0], ngPrc, ngraph::helpers::ActivationTypes::Relu);

    auto input_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
        ngraph::Shape{inputShape.size()}, std::vector<size_t>(inputShape));
    auto input = std::make_shared<ngraph::op::v1::Reshape>(input_relu, input_reshape_pattern, false);

    auto constant_values = ov::test::utils::generate_float_numbers(total_size, 15.5f, 16.1f);
    auto constant = ngraph::builder::makeConstant(ngPrc, std::vector<size_t>({1, total_size}), constant_values);

    auto first_reshape = std::make_shared<ngraph::op::v1::Reshape>(constant, input_reshape_pattern, false);

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector({first_reshape, input}), axis);

    auto final_reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
        ngraph::Shape{2}, std::vector<size_t>({1, 2 * total_size}));
    auto final_reshape = std::make_shared<ngraph::op::v1::Reshape>(concat, final_reshape_pattern, false);

    auto act = ngraph::builder::makeActivation(final_reshape, ngPrc, ngraph::helpers::ActivationTypes::Relu);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(act)};
    function = std::make_shared<ngraph::Function>(results, params, "trivial_concat");
}
}  // namespace SubgraphTestsDefinitions
