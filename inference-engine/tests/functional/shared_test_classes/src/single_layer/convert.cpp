// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convert.hpp"

namespace LayerTestsDefinitions {

std::string ConvertLayerTest::getTestCaseName(const testing::TestParamInfo<ConvertParamsTuple> &obj) {
    InferenceEngine::Precision inputPrecision, targetPrecision;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetName;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(inputShape, inputPrecision, targetPrecision, inLayout, outLayout, targetName) = obj.param;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "targetPRC=" << targetPrecision.name() << "_";
    result << "inputPRC=" << inputPrecision.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetName;
    return result.str();
}

void ConvertLayerTest::SetUp() {
    InferenceEngine::Precision inputPrecision, targetPrecision;
    std::vector<std::vector<size_t>> inputShape;
    std::tie(inputShape, inputPrecision, targetPrecision, inLayout, outLayout, targetDevice) = GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(inputPrecision);
    auto targetPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(targetPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, inputShape);
    auto convert = std::make_shared<ngraph::opset3::Convert>(params.front(), targetPrc);
    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(convert)};
    function = std::make_shared<ngraph::Function>(results, params, "Convert");
}
}  // namespace LayerTestsDefinitions