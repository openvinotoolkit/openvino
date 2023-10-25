// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include "shared_test_classes/subgraph/softsign.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string SoftsignTest::getTestCaseName(const testing::TestParamInfo<softsignParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShape;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void SoftsignTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto abs = std::make_shared<ngraph::op::Abs>(params[0]);

    auto const_1 = ngraph::opset1::Constant::create(ngPrc, ngraph::Shape{}, {1});
    auto const_neg_1 = ngraph::opset1::Constant::create(ngPrc, ngraph::Shape{}, {-1});

    auto add = std::make_shared<ngraph::opset6::Add>(abs, const_1);
    auto power = std::make_shared<ngraph::opset6::Power>(add, const_neg_1);

    auto mul = std::make_shared<ngraph::op::v1::Multiply>(power, params[0]);
    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(mul) };
    function = std::make_shared<ngraph::Function>(results, params, "SoftSignTest");
}

void SoftsignTest::Run() {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    LoadNetwork();
    GenerateInputs();
    Infer();

    function = GenerateNgraphFriendlySoftSign();
    Validate();
}

std::shared_ptr<ngraph::Function> SoftsignTest::GenerateNgraphFriendlySoftSign() {
    InferenceEngine::Precision netPrecision = std::get<0>(this->GetParam());
    std::vector<size_t> inputShape = std::get<3>(this->GetParam());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto abs = std::make_shared<ngraph::op::Abs>(params[0]);
    auto constant_0 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, { 1 });
    auto add = std::make_shared<ngraph::op::v1::Add>(abs, constant_0);
    auto constant_1 = ngraph::builder::makeConstant<float>(ngPrc, inputShape, { -1 });
    auto power = std::make_shared<ngraph::op::v1::Power>(add, constant_1);
    auto mul = std::make_shared<ngraph::op::v1::Multiply>(power, params[0]);

    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(mul) };
    return std::make_shared<ngraph::Function>(results, params, "SoftSignTest");
}
}  // namespace SubgraphTestsDefinitions
