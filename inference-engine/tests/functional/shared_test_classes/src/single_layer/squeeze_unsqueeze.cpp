// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/squeeze_unsqueeze.hpp"

namespace LayerTestsDefinitions {
std::string SqueezeUnsqueezeLayerTest::getTestCaseName(testing::TestParamInfo<squeezeParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    ShapeAxesVec shapeItem;
    std::string targetDevice;
    ngraph::helpers::SqueezeOpType opType;
    std::tie(shapeItem, opType, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = obj.param;
    auto inputShapes = shapeItem[0];

    std::ostringstream result;
    const char separator = '_';
    result << "OpType=" << opType << separator;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "Axes=" << (shapeItem.size() > 1 ? CommonTestUtils::vec2str(shapeItem[1]) : "default") << separator;
    result << "netPRC=" << netPrecision.name() << separator;
    result << "inPRC=" << inPrc.name() << separator;
    result << "outPRC=" << outPrc.name() << separator;
    result << "inL=" << inLayout << separator;
    result << "outL=" << outLayout << separator;
    result << "trgDev=" << targetDevice;
    return result.str();
}

void SqueezeUnsqueezeLayerTest::SetUp() {
    InferenceEngine::Precision netPrecision;
    std::vector<size_t> inputShapes;
    std::vector<int> axesVector;
    ShapeAxesVec shapeItem;
    ngraph::helpers::SqueezeOpType opType;
    std::tie(shapeItem, opType, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    inputShapes = std::vector<size_t>(begin(shapeItem[0]), end(shapeItem[0]));
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    auto params = ngraph::builder::makeParams(ngPrc, {inputShapes});
    std::shared_ptr<ngraph::Node> squeeze;

    if (shapeItem.size() > 1) {
        axesVector = std::vector<int>(begin(shapeItem[1]), end(shapeItem[1]));
        squeeze = ngraph::builder::makeSqueezeUnsqueeze(params.front(), ngraph::element::i64, axesVector, opType);
    } else {
        squeeze = ngraph::builder::makeSqueezeNoAxes(params.front(), ngraph::element::i64);
    }
    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(squeeze)};
    function = std::make_shared<ngraph::Function>(results, params, "Squeeze");
}
} // namespace LayerTestsDefinitions