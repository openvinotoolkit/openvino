// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/prior_box_clustered.hpp"

namespace LayerTestsDefinitions {
using namespace ov::test;

std::string PriorBoxClusteredLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj) {
    ov::test::ElementType netPrecision;
    ov::test::ElementType inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    ov::test::InputShape inputShapes, imageShapes;
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

    result << "IS="      << inputShapes << separator;
    result << "imageS="  << imageShapes << separator;
    result << "netPRC="  << netPrecision << separator;
    result << "inPRC="   << inPrc << separator;
    result << "outPRC="  << outPrc << separator;
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

    InferenceEngine::Layout inLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout outLayout = InferenceEngine::Layout::ANY;
    ov::test::ElementType inPrc = ov::test::ElementType::undefined;
    ov::test::ElementType outPrc = ov::test::ElementType::undefined;
    std::tie(specParams, netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes, imageShapes, targetDevice) = GetParam();

    init_input_shapes({ inputShapes, imageShapes });

    std::tie(widths,
        heights,
        clip,
        step_width,
        step_height,
        step,
        offset,
        variances) = specParams;

    auto params = ngraph::builder::makeDynamicParams(netPrecision, { inputShapes.first, imageShapes.first });

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
