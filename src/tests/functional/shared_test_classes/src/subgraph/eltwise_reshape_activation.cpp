// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/eltwise_reshape_activation.hpp"

namespace SubgraphTestsDefinitions {

using namespace ov::test::utils;
using namespace InferenceEngine;

std::string EltwiseReshapeActivation::getTestCaseName(const testing::TestParamInfo<ParamType>& obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<std::vector<size_t>> shapes;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(shapes, netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(shapes[0]) << "_";
    result << "AS=" << ov::test::utils::vec2str(shapes[1]) << "_";
    result << "PRC=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void EltwiseReshapeActivation::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::vector<std::vector<size_t>> shapes;
    std::map<std::string, std::string> config;
    std::tie(shapes, netPrecision, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shapes[0])),
                              std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(shapes[0]))};
    auto eltw = ngraph::builder::makeEltwise(input[0], input[1], ngraph::helpers::EltwiseTypes::ADD);

    auto reshape_pattern1 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{shapes[1].size()}, shapes[1]);
    auto reshape1 = std::make_shared<ngraph::op::v1::Reshape>(eltw, reshape_pattern1, false);

    auto relu = ngraph::builder::makeActivation(reshape1, ngPrc, ngraph::helpers::ActivationTypes::Relu);

    auto reshape_pattern2 = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{shapes[0].size()}, shapes[0]);
    auto reshape2 = std::make_shared<ngraph::op::v1::Reshape>(relu, reshape_pattern2, false);

    function = std::make_shared<ngraph::Function>(reshape2, input, "EltwiseReshapeActivation");
}
}  // namespace SubgraphTestsDefinitions
