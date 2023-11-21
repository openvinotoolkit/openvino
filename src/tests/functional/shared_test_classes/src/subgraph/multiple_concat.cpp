// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_models/builders.hpp"
#include "shared_test_classes/subgraph/multiple_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string MultipleConcatTest::getTestCaseName(const testing::TestParamInfo<multipleConcatParams> &obj) {
    std::string targetDevice;
    InferenceEngine::Precision netPrecision;
    size_t inputSize;
    size_t constantSize;
    std::map<std::string, std::string> config;
    std::tie(targetDevice, netPrecision, inputSize, constantSize, config) = obj.param;
    std::ostringstream result;

    result << "netPrecision=" << netPrecision.name() << "_";
    result << "IS=" << inputSize << "_";
    result << "CS=" << constantSize << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : config) {
        result << "_configItem=" << configItem.second;
    }
    return result.str();
}

void MultipleConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> config;
    size_t inputSize;
    size_t constantSize;
    std::tie(targetDevice, netPrecision, inputSize, constantSize, config) = this->GetParam();
    configuration.insert(config.begin(), config.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    std::vector<size_t> input_dims { 1, inputSize };
    std::vector<size_t> constant_dims {1, constantSize};

    auto concat_1_vals = ov::test::utils::generate_float_numbers(constantSize, -2.0f, 2.0f);
    auto concat_2_vals = ov::test::utils::generate_float_numbers(constantSize, -5.0f, 5.0f);

    ov::ParameterVector input_parameter {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(input_dims))};

    auto const_1 = ngraph::builder::makeConstant(ngPrc, constant_dims, concat_1_vals);
    auto concat_1 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{const_1, input_parameter[0]}, 1);

    auto const_2 = ngraph::builder::makeConstant(ngPrc, constant_dims, concat_1_vals);
    auto concat_2 = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{concat_1, const_2}, 1);

    auto act = ngraph::builder::makeActivation(concat_2, ngPrc, ngraph::helpers::ActivationTypes::Relu);

    function = std::make_shared<ngraph::Function>(act, input_parameter, "multiple_concat");
}
}  // namespace SubgraphTestsDefinitions
