// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/activation_concats_eltwise.hpp"

namespace SubgraphTestsDefinitions {

using namespace ov::test::utils;
using namespace InferenceEngine;

std::string ActivationConcatsEltwise::getTestCaseName(const testing::TestParamInfo<ParamType>& obj) {
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    size_t concatSize;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(inputSize, concatSize, netPrecision, targetDevice, configuration) = obj.param;

    std::ostringstream result;
    result << "IS=" << inputSize << "_";
    result << "CS=" << concatSize << "_";
    result << "PRC=" << netPrecision.name() << "_";
    result << "dev=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void ActivationConcatsEltwise::SetUp() {
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    size_t concatSize;
    std::map<std::string, std::string> config;
    std::tie(inputSize, concatSize, netPrecision, targetDevice, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector input{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape{1, inputSize})};

    auto relu = ngraph::builder::makeActivation(input[0], ngPrc, ngraph::helpers::ActivationTypes::Relu);

    auto concat_vals_1 = ov::test::utils::generate_float_numbers(concatSize, 14, 14);
    auto concat_vals_2 = ov::test::utils::generate_float_numbers(concatSize, 14, 14);
    auto concat_const_1 = ngraph::builder::makeConstant(ngPrc, {1, concatSize}, concat_vals_1);
    auto concat_const_2 = ngraph::builder::makeConstant(ngPrc, {1, concatSize}, concat_vals_2);

    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concat_const_1, relu}, 1);
    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concat_const_2, relu}, 1);

    auto eltw = ngraph::builder::makeEltwise(concat_1, concat_2, ngraph::helpers::EltwiseTypes::ADD);

    auto reshape_pattern = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{2}, std::vector<size_t>({1, inputSize + concatSize}));
    auto final_reshape = std::make_shared<ngraph::op::v1::Reshape>(eltw, reshape_pattern, false);
    function = std::make_shared<ngraph::Function>(final_reshape, input, "ActivationConcatsEltwise");
}
}  // namespace SubgraphTestsDefinitions
