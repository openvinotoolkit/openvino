// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/prior_box_clustered.hpp"

namespace LayerTestsDefinitions {
std::string PriorBoxClusteredLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, imageShapes;
    std::string targetDevice;
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams,
        netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes,
        imageShapes,
        targetDevice) = obj.param;

    std::vector<float> widths, heights, variances;
    float step_width, step_height, step, offset;
    bool clip;
    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        step,
        offset,
        variances) = specParams;

    std::ostringstream result;
    const char separator = '_';

    result << "IS="      << CommonTestUtils::vec2str(inputShapes) << separator;
    result << "imageS="  << CommonTestUtils::vec2str(imageShapes) << separator;
    result << "netPRC="  << netPrecision.name()   << separator;
    result << "inPRC="   << inPrc.name() << separator;
    result << "outPRC="  << outPrc.name() << separator;
    result << "inL="     << inLayout << separator;
    result << "outL="    << outLayout << separator;
    result << "widths="  << CommonTestUtils::vec2str(widths)  << separator;
    result << "heights=" << CommonTestUtils::vec2str(heights) << separator;
    result << "variances=";
    if (variances.empty())
        result << "()" << separator;
    else
        result << CommonTestUtils::vec2str(variances) << separator;
    result << "stepWidth="  << step_width  << separator;
    result << "stepHeight=" << step_height << separator;
    result << "step="       << step << separator;
    result << "offset="     << offset      << separator;
    result << "clip="       << std::boolalpha << clip << separator;
    result << "trgDev="     << targetDevice;
    return result.str();
}

void PriorBoxClusteredLayerTest::SetUp() {
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams, netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes, imageShapes, targetDevice) = GetParam();

    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        step,
        offset,
        variances) = specParams;

    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, { inputShapes, imageShapes });

    ngraph::op::PriorBoxClusteredAttrs attributes;
    attributes.widths = widths;
    attributes.heights = heights;
    attributes.clip = clip;
    attributes.step_widths = step_width;
    attributes.step_heights = step_height;
    attributes.step = step;
    attributes.offset = offset;
    attributes.variances = variances;

    auto shape_of_1 = std::make_shared<ngraph::opset3::ShapeOf>(params[0]);
    auto shape_of_2 = std::make_shared<ngraph::opset3::ShapeOf>(params[1]);
    auto priorBoxClustered = std::make_shared<ngraph::op::PriorBoxClustered>(
        shape_of_1,
        shape_of_2,
        attributes);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(priorBoxClustered) };
    function = std::make_shared<ngraph::Function>(results, params, "PB_Clustered");
}
}  // namespace LayerTestsDefinitions
