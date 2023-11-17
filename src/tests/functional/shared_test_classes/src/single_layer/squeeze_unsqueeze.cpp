// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/squeeze_unsqueeze.hpp"

namespace LayerTestsDefinitions {
std::string SqueezeUnsqueezeLayerTest::getTestCaseName(const testing::TestParamInfo<squeezeParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    ShapeAxesTuple shapeItem;
    std::string targetDevice;
    ngraph::helpers::SqueezeOpType opType;
    std::tie(shapeItem, opType, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = obj.param;

    std::ostringstream result;
    const char separator = '_';
    result << "OpType=" << opType << separator;
    result << "IS=" << ov::test::utils::vec2str(shapeItem.first) << separator;
    result << "Axes=" << (shapeItem.second.empty() ? "default" : ov::test::utils::vec2str(shapeItem.second)) << separator;
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
    ShapeAxesTuple shapeItem;
    ngraph::helpers::SqueezeOpType opType;
    std::tie(shapeItem, opType, netPrecision, inPrc, outPrc, inLayout, outLayout, targetDevice) = GetParam();
    std::tie(inputShapes, axesVector) = shapeItem;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ov::ParameterVector params {std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShapes))};
    std::shared_ptr<ngraph::Node> op;

    if (axesVector.empty() && opType == ngraph::helpers::SqueezeOpType::SQUEEZE) {
        op = std::make_shared<ngraph::opset1::Squeeze>(params.front());
    } else {
        OPENVINO_SUPPRESS_DEPRECATED_START
        op = ngraph::builder::makeSqueezeUnsqueeze(params.front(), ngraph::element::i64, axesVector, opType);
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(op)};
    function = std::make_shared<ngraph::Function>(results, params, "Squeeze");
}
} // namespace LayerTestsDefinitions
