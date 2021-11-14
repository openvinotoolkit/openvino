// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/prior_box.hpp"

namespace LayerTestDefinitions {
using namespace ov::test;

std::string PriorBoxLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxLayerParams>& obj) {
    ov::test::ElementType netPrecision;
    ov::test::ElementType inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    ov::test::InputShape inputShapes;
    ov::test::InputShape imageShapes;
    std::string targetDevice;
    priorBoxSpecificParams specParams;
    std::tie(specParams,
        netPrecision,
        inPrc, outPrc, inLayout, outLayout,
        inputShapes,
        imageShapes,
        targetDevice) = obj.param;

    std::vector<float> min_size, max_size, aspect_ratio, density, fixed_ratio, fixed_size, variance;
    float step, offset;
    bool clip, flip, scale_all_sizes, min_max_aspect_ratios_order;
    std::tie(min_size, max_size, aspect_ratio,
             density, fixed_ratio, fixed_size, clip,
             flip, step, offset, variance, scale_all_sizes, min_max_aspect_ratios_order) = specParams;

    std::ostringstream result;
    const char separator = '_';
    result << "IS="      << inputShapes << separator;
    result << "imageS="  << imageShapes << separator;
    result << "netPRC="  << netPrecision   << separator;
    result << "inPRC="   << inPrc << separator;
    result << "outPRC="  << outPrc << separator;
    result << "inL="     << inLayout << separator;
    result << "outL="    << outLayout << separator;
    result << "min_s=" << CommonTestUtils::vec2str(min_size) << separator;
    result << "max_s=" << CommonTestUtils::vec2str(max_size)<< separator;
    result << "asp_r=" << CommonTestUtils::vec2str(aspect_ratio)<< separator;
    result << "dens=" << CommonTestUtils::vec2str(density)<< separator;
    result << "fix_r=" << CommonTestUtils::vec2str(fixed_ratio)<< separator;
    result << "fix_s=" << CommonTestUtils::vec2str(fixed_size)<< separator;
    result << "var=" << CommonTestUtils::vec2str(variance)<< separator;
    result << "step=" << step << separator;
    result << "off=" << offset << separator;
    result << "clip=" << clip << separator;
    result << "flip=" << flip<< separator;
    result << "scale_all=" << scale_all_sizes << separator;
    result << "min_max_aspect_ratios_order=" << min_max_aspect_ratios_order << separator;
    result << "trgDev=" << targetDevice;

    return result.str();
}

void PriorBoxLayerTest::SetUp() {
    priorBoxSpecificParams specParams;

    InferenceEngine::Layout inLayout = InferenceEngine::Layout::ANY;
    InferenceEngine::Layout outLayout = InferenceEngine::Layout::ANY;
    ov::test::ElementType inPrc = ov::test::ElementType::undefined;
    ov::test::ElementType outPrc = ov::test::ElementType::undefined;
    std::tie(specParams, netPrecision,
             inPrc, outPrc, inLayout, outLayout,
             inputShapes, imageShapes, targetDevice) = GetParam();

    init_input_shapes({ inputShapes, imageShapes });

    std::tie(min_size, max_size, aspect_ratio,
             density, fixed_ratio, fixed_size, clip,
             flip, step, offset, variance, scale_all_sizes,
             min_max_aspect_ratios_order) = specParams;

    auto params = ngraph::builder::makeDynamicParams(netPrecision, {inputShapes.first, imageShapes.first});

    ngraph::op::v8::PriorBox::Attributes attributes;
    attributes.min_size = min_size;
    attributes.max_size = max_size;
    attributes.aspect_ratio = aspect_ratio;
    attributes.density = density;
    attributes.fixed_ratio = fixed_ratio;
    attributes.fixed_size = fixed_size;
    attributes.variance = variance;
    attributes.step = step;
    attributes.offset = offset;
    attributes.clip = clip;
    attributes.flip = flip;
    attributes.min_max_aspect_ratios_order = min_max_aspect_ratios_order;

    auto shape_of_1 = std::make_shared<ngraph::opset3::ShapeOf>(params[0]);
    auto shape_of_2 = std::make_shared<ngraph::opset3::ShapeOf>(params[1]);
    auto priorBox = std::make_shared<ngraph::op::v8::PriorBox>(
        shape_of_1,
        shape_of_2,
        attributes);

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(priorBox)};
    function = std::make_shared <ngraph::Function>(results, params, "PriorBoxFunction");
}
} // namespace LayerTestDefinitions
