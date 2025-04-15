// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/prior_box.hpp"

#include "openvino/pass/constant_folding.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/prior_box_clustered.hpp"

namespace ov {
namespace test {
std::string PriorBoxLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxLayerParams>& obj) {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string target_device;
    priorBoxSpecificParams spec_params;
    std::tie(spec_params, model_type, shapes, target_device) = obj.param;

    std::vector<float> min_size, max_size, aspect_ratio, density, fixed_ratio, fixed_size, variance;
    float step, offset;
    bool clip, flip, scale_all_sizes, min_max_aspect_ratios_order;
    std::tie(min_size, max_size, aspect_ratio, density, fixed_ratio, fixed_size, clip,
             flip, step, offset, variance, scale_all_sizes, min_max_aspect_ratios_order) = spec_params;

    std::ostringstream result;
    const char separator = '_';
    result << "IS=(";
    for (size_t i = 0lu; i < shapes.size(); i++) {
        result << ov::test::utils::partialShape2str({shapes[i].first}) << (i < shapes.size() - 1lu ? "_" : "");
    }
    result << ")_TS=";
    for (size_t i = 0lu; i < shapes.front().second.size(); i++) {
        result << "{";
        for (size_t j = 0lu; j < shapes.size(); j++) {
            result << ov::test::utils::vec2str(shapes[j].second[i]) << (j < shapes.size() - 1lu ? "_" : "");
        }
        result << "}_";
    }
    result << "netPRC="  << model_type.get_type_name()   << separator;
    result << "min_s=" << ov::test::utils::vec2str(min_size) << separator;
    result << "max_s=" << ov::test::utils::vec2str(max_size)<< separator;
    result << "asp_r=" << ov::test::utils::vec2str(aspect_ratio)<< separator;
    result << "dens=" << ov::test::utils::vec2str(density)<< separator;
    result << "fix_r=" << ov::test::utils::vec2str(fixed_ratio)<< separator;
    result << "fix_s=" << ov::test::utils::vec2str(fixed_size)<< separator;
    result << "var=" << ov::test::utils::vec2str(variance)<< separator;
    result << "step=" << step << separator;
    result << "off=" << offset << separator;
    result << "clip=" << clip << separator;
    result << "flip=" << flip<< separator;
    result << "scale_all=" << scale_all_sizes << separator;
    result << "min_max_aspect_ratios_order=" << min_max_aspect_ratios_order << separator;
    result << "trgDev=" << target_device;

    return result.str();
}

void PriorBoxLayerTest::SetUp() {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::vector<float> min_size;
    std::vector<float> max_size;
    std::vector<float> aspect_ratio;
    std::vector<float> density;
    std::vector<float> fixed_ratio;
    std::vector<float> fixed_size;
    std::vector<float> variance;
    float step;
    float offset;
    bool clip;
    bool flip;
    bool scale_all_sizes;
    bool min_max_aspect_ratios_order;

    priorBoxSpecificParams spec_params;
    std::tie(spec_params, model_type, shapes, targetDevice) = GetParam();

    std::tie(min_size, max_size, aspect_ratio, density, fixed_ratio, fixed_size, clip,
             flip, step, offset, variance, scale_all_sizes, min_max_aspect_ratios_order) = spec_params;
    init_input_shapes(shapes);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1])};

    ov::op::v8::PriorBox::Attributes attributes;
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
    attributes.scale_all_sizes = scale_all_sizes;
    attributes.min_max_aspect_ratios_order = min_max_aspect_ratios_order;

    auto shape_of_1 = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
    auto shape_of_2 = std::make_shared<ov::op::v3::ShapeOf>(params[1]);
    auto priorBox = std::make_shared<ov::op::v8::PriorBox>(
        shape_of_1,
        shape_of_2,
        attributes);

    ov::pass::disable_constant_folding(priorBox);

    auto result = std::make_shared<ov::op::v0::Result>(priorBox);
    function = std::make_shared <ov::Model>(result, params, "PriorBoxFunction");
}
} // namespace test
} // namespace ov
