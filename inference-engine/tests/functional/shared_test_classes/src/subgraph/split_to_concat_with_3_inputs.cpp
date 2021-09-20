// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_to_concat_with_3_inputs.hpp"
#include "ngraph_functions/builders.hpp"

namespace SubgraphTestsDefinitions {

std::string SplitConcatWith3InputsTest::getTestCaseName(testing::TestParamInfo<SplitConcatWith3InputsParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    std::map<std::string, std::string> configuration;
    std::vector<size_t> inputShape;
    std::tie(netPrecision, targetDevice, configuration, inputShape) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    for (auto const& configItem : configuration) {
        result << "_configItem=" << configItem.first << "_" << configItem.second;
    }
    return result.str();
}

void SplitConcatWith3InputsTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::map<std::string, std::string> tempConfig;
    std::vector<size_t> inputShape;
    std::tie(netPrecision, targetDevice, tempConfig, inputShape) = this->GetParam();
    configuration.insert(tempConfig.begin(), tempConfig.end());
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, { inputShape });

    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 3, 1);
    auto relu1 = std::make_shared<ngraph::opset3::Relu>(split->output(0));
    auto tanh1 = std::make_shared<ngraph::opset3::Tanh>(split->output(1));

    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1, tanh1, split->output(2)}, 1);
    auto relu2 = std::make_shared<ngraph::opset3::Relu>(concat);

    function = std::make_shared<ngraph::Function>(relu2, params, "SplitConcatWith3InputsTest");
}
}  // namespace SubgraphTestsDefinitions