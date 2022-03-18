// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/conversion.hpp"

#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string ConversionLayerTest::getTestCaseName(const testing::TestParamInfo<ConversionParamsTuple>& obj) {
    ngraph::helpers::ConversionTypes conversionOpType;
    InferenceEngine::Precision inputPrecision, targetPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(conversionOpType, inputShape, inputPrecision, targetPrecision, inLayout, outLayout, targetName) =
        obj.param;
    std::ostringstream result;
    result << "conversionOpType=" << conversionNames[conversionOpType] << "_";
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "inputPRC=" << inputPrecision.name() << "_";
    result << "targetPRC=" << targetPrecision.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConversionLayerTest::SetUp() {
    if (FuncTestUtils::SkipTestsConfig::currentTestIsDisabled()) {
        GTEST_SKIP() << "Disabled test due to configuration" << std::endl;
    }
    ngraph::helpers::ConversionTypes conversionOpType;
    InferenceEngine::Precision inputPrecision, targetPrecision;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(conversionOpType, inputShape, inputPrecision, targetPrecision, inLayout, outLayout.front(), targetDevice) =
        GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto targetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(targetPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, inputShape);
    auto conversion = ngraph::builder::makeConversion(params.front(), targetPrc, conversionOpType);

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(conversion)};
    function = std::make_shared<ngraph::Function>(results, params, "Conversion");
}
}  // namespace LayerTestsDefinitions
