// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_plugin_config.hpp>
#include "shared_test_classes/single_layer/interpolate.hpp"
#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"

using ngraph::helpers::operator<<;

namespace LayerTestsDefinitions {

std::string InterpolateLayerTest::getTestCaseName(testing::TestParamInfo<InterpolateLayerTestParams> obj) {
    InterpolateSpecificParams interpolateParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, targetShapes;
    std::string targetDevice;
    std::map<std::string, std::string> additional_config;
    std::tie(interpolateParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetShapes, targetDevice, additional_config) = obj.param;
    std::vector<size_t> padBegin, padEnd;
    std::vector<int64_t> axes;
    std::vector<float> scales;
    bool antialias;
    ngraph::op::v4::Interpolate::InterpolateMode mode;
    ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
    ngraph::op::v4::Interpolate::NearestMode nearestMode;
    double cubeCoef;
    std::tie(mode, shapeCalcMode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef, axes, scales) = interpolateParams;
    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "TS=" << CommonTestUtils::vec2str(targetShapes) << "_";
    result << "InterpolateMode=" << mode << "_";
    result << "ShapeCalcMode=" << shapeCalcMode << "_";
    result << "CoordinateTransformMode=" << coordinateTransformMode << "_";
    result << "NearestMode=" << nearestMode << "_";
    result << "CubeCoef=" << cubeCoef << "_";
    result << "Antialias=" << antialias << "_";
    result << "PB=" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE=" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "Axes=" << CommonTestUtils::vec2str(axes) << "_";
    result << "Scales=" << CommonTestUtils::vec2str(scales) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void InterpolateLayerTest::SetUp() {
    InterpolateSpecificParams interpolateParams;
    std::vector<size_t> inputShape, targetShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::map<std::string, std::string> additional_config;
    std::tie(interpolateParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShape, targetDevice, additional_config) = this->GetParam();
    std::vector<size_t> padBegin, padEnd;
    std::vector<int64_t> axes;
    std::vector<float> scales;
    bool antialias;
    ngraph::op::v4::Interpolate::InterpolateMode mode;
    ngraph::op::v4::Interpolate::ShapeCalcMode shapeCalcMode;
    ngraph::op::v4::Interpolate::CoordinateTransformMode coordinateTransformMode;
    ngraph::op::v4::Interpolate::NearestMode nearestMode;

    configuration.insert(additional_config.begin(), additional_config.end());

    double cubeCoef;
    std::tie(mode, shapeCalcMode, coordinateTransformMode, nearestMode, antialias, padBegin, padEnd, cubeCoef, axes, scales) = interpolateParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto sizesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {targetShape.size()}, targetShape);
    auto sizesInput = std::make_shared<ngraph::opset3::Constant>(sizesConst);

    auto scales_const = ngraph::opset3::Constant(ngraph::element::Type_t::f32, {scales.size()}, scales);
    auto scalesInput = std::make_shared<ngraph::opset3::Constant>(scales_const);

    auto axesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {axes.size()}, axes);
    auto axesInput = std::make_shared<ngraph::opset3::Constant>(axesConst);

    ngraph::op::v4::Interpolate::InterpolateAttrs interpolateAttributes{mode, shapeCalcMode, padBegin,
        padEnd, coordinateTransformMode, nearestMode, antialias, cubeCoef};
    auto interpolate = std::make_shared<ngraph::op::v4::Interpolate>(params[0],
                                                                     sizesInput,
                                                                     scalesInput,
                                                                     axesInput,
                                                                     interpolateAttributes);
    const ngraph::ResultVector results{std::make_shared<ngraph::opset3::Result>(interpolate)};
    function = std::make_shared<ngraph::Function>(results, params, "interpolate");
}

//Interpolate-1

std::string Interpolate1LayerTest::getTestCaseName(testing::TestParamInfo<Interpolate1LayerTestParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision dataPrecision;
    InferenceEngine::Layout dataLayout;
    InferenceEngine::SizeVector inputShapes, targetShapes;
    std::string mode;
    ngraph::AxisSet axes;
    bool antialias;
    std::vector<size_t> pads;
    std::string targetDevice;
    std::tie(netPrecision, dataPrecision, dataLayout, inputShapes, targetShapes,
             mode, axes, antialias, pads, targetDevice) = obj.param;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "TS=" << CommonTestUtils::vec2str(targetShapes) << "_";
    result << "InterpolateMode=" << mode << "_";
    result << "Antialias=" << antialias << "_";
    result << "PB=" << CommonTestUtils::vec2str(pads) << "_";
    result << "PE=" << CommonTestUtils::vec2str(pads) << "_";
    result << "Axes=" << CommonTestUtils::vec2str(axes.to_vector()) << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "PRC=" << dataPrecision.name() << "_";
    result << "Layout=" << dataLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void Interpolate1LayerTest::SetUp() {
    std::vector<size_t> inputShape, targetShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    InferenceEngine::Precision dataPrecision;
    InferenceEngine::Layout dataLayout;
    std::string mode;
    ngraph::AxisSet axes;
    bool antialias;
    std::vector<size_t> pads;
    std::tie(netPrecision, dataPrecision, dataLayout, inputShape, targetShape,
             mode, axes, antialias, pads, targetDevice) = this->GetParam();

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});

    auto sizesConst = ngraph::opset3::Constant(ngraph::element::Type_t::i64, {targetShape.size()}, targetShape);
    auto sizesInput = std::make_shared<ngraph::opset3::Constant>(sizesConst);

    bool align_corners = true;

    ngraph::op::v0::InterpolateAttrs interpolateAttributes;
    interpolateAttributes.axes = axes;
    interpolateAttributes.mode = mode;
    interpolateAttributes.align_corners = align_corners;
    interpolateAttributes.antialias = antialias;
    interpolateAttributes.pads_begin = pads;
    interpolateAttributes.pads_end = pads;

    auto interpolate = std::make_shared<ngraph::op::v0::Interpolate>(params[0], sizesInput, interpolateAttributes);

    const ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(interpolate)};
    function = std::make_shared<ngraph::Function>(results, params, "interpolate");
}

}  // namespace LayerTestsDefinitions
