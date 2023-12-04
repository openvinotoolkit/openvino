// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/pooling.hpp"

namespace LayerTestsDefinitions {

std::string PoolingLayerTest::getTestCaseName(const testing::TestParamInfo<poolLayerTestParamsSet>& obj) {
    poolSpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<size_t> inputShapes;
    std::string targetDevice;
    std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    switch (poolType) {
        case ngraph::helpers::PoolingTypes::MAX:
            result << "MaxPool_";
            break;
        case ngraph::helpers::PoolingTypes::AVG:
            result << "AvgPool_";
            result << "ExcludePad=" << excludePad << "_";
            break;
    }
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "Rounding=" << roundingType << "_";
    result << "AutoPad=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

std::string GlobalPoolingLayerTest::getTestCaseName(const testing::TestParamInfo<globalPoolLayerTestParamsSet>& obj) {
    poolSpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::string targetDevice;
    size_t channels;
    std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, channels, targetDevice) = obj.param;
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    std::vector<size_t> inputShapes = {1, channels, kernel[0], kernel[1]};

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    switch (poolType) {
        case ngraph::helpers::PoolingTypes::MAX:
            result << "MaxPool_";
            break;
        case ngraph::helpers::PoolingTypes::AVG:
            result << "AvgPool_";
            result << "ExcludePad=" << excludePad << "_";
            break;
    }
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    if (padType == ngraph::op::PadType::EXPLICIT) {
        result << "Rounding=" << roundingType << "_";
    }
    result << "AutoPad=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

std::string MaxPoolingV8LayerTest::getTestCaseName(const testing::TestParamInfo<maxPoolV8LayerTestParamsSet>& obj) {
    maxPoolV8SpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<size_t> inputShapes;
    std::string targetDevice;
    std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    ngraph::element::Type indexElementType;
    int64_t axis;
    std::tie(kernel, stride, dilation, padBegin, padEnd, indexElementType, axis, roundingType, padType) = poolParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "D" << ov::test::utils::vec2str(dilation) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "IET" << indexElementType << "_";
    result << "A" << axis << "_";
    result << "Rounding=" << roundingType << "_";
    result << "AutoPad=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void PoolingLayerTest::SetUp() {
    poolSpecificParams poolParams;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    OPENVINO_SUPPRESS_DEPRECATED_START
    std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(params[0],
                                                                         stride,
                                                                         padBegin,
                                                                         padEnd,
                                                                         kernel,
                                                                         roundingType,
                                                                         padType,
                                                                         excludePad,
                                                                         poolType);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pooling)};
    function = std::make_shared<ngraph::Function>(results, params, "pooling");
}

void GlobalPoolingLayerTest::SetUp() {
    poolSpecificParams poolParams;
    InferenceEngine::Precision netPrecision;
    size_t channels;
    std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, channels, targetDevice) = this->GetParam();
    ngraph::helpers::PoolingTypes poolType;
    std::vector<size_t> kernel, stride;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    bool excludePad;
    std::tie(poolType, kernel, stride, padBegin, padEnd, roundingType, padType, excludePad) = poolParams;

    std::vector<size_t> inputShape = {1, channels, kernel[1], kernel[0]};

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    OPENVINO_SUPPRESS_DEPRECATED_START
    std::shared_ptr<ngraph::Node> pooling = ngraph::builder::makePooling(params[0],
                                                                         stride,
                                                                         padBegin,
                                                                         padEnd,
                                                                         kernel,
                                                                         roundingType,
                                                                         padType,
                                                                         excludePad,
                                                                         poolType);
    OPENVINO_SUPPRESS_DEPRECATED_END

    ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(pooling)};
    function = std::make_shared<ngraph::Function>(results, params, "pooling");
}

void MaxPoolingV8LayerTest::SetUp() {
    maxPoolV8SpecificParams poolParams;
    std::vector<size_t> inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(poolParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
    std::vector<size_t> kernel, stride, dilation;
    std::vector<size_t> padBegin, padEnd;
    ngraph::op::PadType padType;
    ngraph::op::RoundingType roundingType;
    ngraph::element::Type indexElementType;
    int64_t axis;
    std::tie(kernel, stride, dilation, padBegin, padEnd, indexElementType, axis, roundingType, padType) = poolParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    auto maxPool = std::make_shared<ov::op::v8::MaxPool>(params[0], stride, dilation, padBegin, padEnd,
                                                         kernel, roundingType, padType,
                                                         indexElementType, axis);

    const auto maxPoolV8_second_output_is_supported = targetDevice == ov::test::utils::DEVICE_GPU;
    ngraph::ResultVector results;
    if (maxPoolV8_second_output_is_supported) {
        results = {std::make_shared<ngraph::opset3::Result>(maxPool->output(0)),
                   std::make_shared<ngraph::opset3::Result>(maxPool->output(1))};
    } else {
        results = { std::make_shared<ngraph::opset3::Result>(maxPool->output(0)) };
    }
    function = std::make_shared<ngraph::Function>(results, params, "MaxPoolV8");
}

}  // namespace LayerTestsDefinitions
