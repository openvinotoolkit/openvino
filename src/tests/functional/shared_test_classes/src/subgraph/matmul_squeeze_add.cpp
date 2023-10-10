// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/matmul_squeeze_add.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string MatmulSqueezeAddTest::getTestCaseName(const testing::TestParamInfo<matmulSqueezeAddParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShape;
    std::size_t outputSize;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration, inputShape, outputSize) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "OS=" << outputSize << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void MatmulSqueezeAddTest::SetUp() {
    auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    size_t outputSize;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape, outputSize) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto constant_0 = ngraph::builder::makeConstant<float>(ngPrc, { outputSize, inputShape[1] },
        ov::test::utils::generate_float_numbers(outputSize * inputShape[1], 0, 1, seed), false);
    auto matmul_0 = std::make_shared<ngraph::op::MatMul>(params[0], constant_0, false, true);

    auto constant_1 = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 1 }, std::vector<size_t>{0});
    auto unsqueeze_0 = std::make_shared<ngraph::op::v0::Unsqueeze>(matmul_0, constant_1);

    auto constant_2 = ngraph::builder::makeConstant<float>(ngPrc, { 1, inputShape[0], outputSize },
        ov::test::utils::generate_float_numbers(inputShape[0] * outputSize, 0, 1, seed), false);
    auto add_0 = std::make_shared<ngraph::op::v1::Add>(unsqueeze_0, constant_2);

    auto constant_3 = std::make_shared<ngraph::op::Constant>(ngraph::element::Type_t::i64, ngraph::Shape{ 1 }, std::vector<size_t>{0});
    auto squeeze_0 = std::make_shared<ngraph::op::Squeeze>(add_0, constant_3);

    ngraph::ResultVector results {std::make_shared<ngraph::op::Result>(squeeze_0)};
    function = std::make_shared<ngraph::Function>(results, params, "MatmulSqueezeAddTest");
}
}  // namespace SubgraphTestsDefinitions
