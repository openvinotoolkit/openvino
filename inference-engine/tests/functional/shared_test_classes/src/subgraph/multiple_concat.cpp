// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
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

    auto concat_1_vals = CommonTestUtils::generate_float_numbers(constantSize, -2.0f, 2.0f);
    auto concat_2_vals = CommonTestUtils::generate_float_numbers(constantSize, -5.0f, 5.0f);

    auto input_parameter = ngraph::builder::makeParams(ngPrc, {input_dims});

    auto const_1 = ngraph::builder::makeConstant(ngPrc, constant_dims, concat_1_vals);
    auto concat_1 = ngraph::builder::makeConcat({const_1, input_parameter[0]}, 1);

    auto const_2 = ngraph::builder::makeConstant(ngPrc, constant_dims, concat_1_vals);
    auto concat_2 = ngraph::builder::makeConcat({concat_1, const_2}, 1);

    auto act = ngraph::builder::makeActivation(concat_2, ngPrc, ngraph::helpers::ActivationTypes::Relu);

    function = std::make_shared<ngraph::Function>(act, input_parameter, "multiple_concat");
}
}  // namespace SubgraphTestsDefinitions
