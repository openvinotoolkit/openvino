// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/split_conv_concat.hpp"

namespace SubgraphTestsDefinitions {

std::string SplitConvConcat::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes, newInputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void SplitConvConcat::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto split = ngraph::builder::makeSplit(params[0], ngPrc, 2, 1);

    auto conv1 = ngraph::builder::makeConvolution(split->output(0), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu1 = std::make_shared<ngraph::opset1::Relu>(conv1);

    auto conv2 = ngraph::builder::makeConvolution(split->output(1), ngPrc, {3, 3}, {1, 1}, {0, 0}, {0, 0}, {1, 1},
                                                  ngraph::op::PadType::EXPLICIT, 5);
    auto relu2 = std::make_shared<ngraph::opset1::Relu>(conv2);
    auto concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{relu1->output(0), relu2->output(0)}, 1);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, params, "SplitConvConcat");
}

}  // namespace SubgraphTestsDefinitions