// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/nonzero.hpp"

namespace LayerTestsDefinitions {

std::string NonZeroLayerTest::getTestCaseName(testing::TestParamInfo<NonZeroLayerTestParamsSet> obj) {
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

void NonZeroLayerTest::SetUp() {
    auto inputShape     = std::vector<std::size_t>{};
    auto inputPrecision = InferenceEngine::Precision::UNSPECIFIED;
    ConfigMap additionalConfig;
    std::tie(inputShape, inputPrecision, targetDevice, additionalConfig) = GetParam();

    configuration.insert(additionalConfig.cbegin(), additionalConfig.cend());

    const auto& precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    const auto& paramNode = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));

    auto nonZeroOp = std::make_shared<ngraph::opset3::NonZero>(paramNode->output(0));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(nonZeroOp)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{paramNode}, "non_zero");
}
}  // namespace LayerTestsDefinitions
