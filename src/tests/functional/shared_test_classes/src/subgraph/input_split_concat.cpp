// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/input_split_concat.hpp"
#include "ov_models/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string InputSplitConcatTest::getTestCaseName(const testing::TestParamInfo<InputSplitConcatParams>& obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::vector<size_t> inputShape;
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

void InputSplitConcatTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);
    auto relu1 = std::make_shared<ngraph::opset3::Relu>(split->output(0));

    auto const_vals = ov::test::utils::generate_float_numbers(inputShape[1], -5.0f, 5.0f);
    auto constant = ngraph::builder::makeConstant(ngPrc, inputShape, const_vals);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{constant, split->output(1)}, 1);
    auto relu2 = std::make_shared<ngraph::opset3::Relu>(concat);

    ngraph::ResultVector results{ std::make_shared<ngraph::op::Result>(relu1), std::make_shared<ngraph::op::Result>(relu2) };
    function = std::make_shared<ngraph::Function>(results, params, "InputSplitConcatTest");
}
}  // namespace SubgraphTestsDefinitions
