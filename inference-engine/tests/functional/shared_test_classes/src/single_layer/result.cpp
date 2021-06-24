// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/result.hpp"

namespace LayerTestsDefinitions {

std::string ResultLayerTest::getTestCaseName(testing::TestParamInfo<ResultTestParamSet> obj) {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision inputPrecision;
    std::string targetDevice;
    ConfigMap additionalConfig;
    std::tie(inputShape, inputPrecision, targetDevice, additionalConfig) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "inPRC=" << inputPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void ResultLayerTest::SetUp() {
    std::vector<size_t> inputShape;
    InferenceEngine::Precision inputPrecision;
    std::string targetDevice;
    ConfigMap additionalConfig;
    std::tie(inputShape, inputPrecision, targetDevice, additionalConfig) = GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(params[0])};
    function = std::make_shared<ngraph::Function>(results, params, "result");
}
}  // namespace LayerTestsDefinitions
