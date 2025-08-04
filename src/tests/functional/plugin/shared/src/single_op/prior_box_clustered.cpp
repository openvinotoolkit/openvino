// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_op/prior_box_clustered.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/prior_box_clustered.hpp"

namespace ov {
namespace test {
std::string PriorBoxClusteredLayerTest::getTestCaseName(const testing::TestParamInfo<priorBoxClusteredLayerParams>& obj) {
    ov::element::Type model_type;
    std::vector<InputShape> shapes;
    std::string target_device;
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams, model_type, shapes, target_device) = obj.param;

    std::vector<float> widths, heights, variances;
    float step_width, step_height, step, offset;
    bool clip;
    std::tie(widths, heights, clip, step_width, step_height, step, offset, variances) = specParams;

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
    result << "widths="  << ov::test::utils::vec2str(widths)  << separator;
    result << "heights=" << ov::test::utils::vec2str(heights) << separator;
    result << "variances=";
    if (variances.empty())
        result << "()" << separator;
    else
        result << ov::test::utils::vec2str(variances) << separator;
    result << "stepWidth="  << step_width  << separator;
    result << "stepHeight=" << step_height << separator;
    result << "step="       << step << separator;
    result << "offset="     << offset      << separator;
    result << "clip="       << std::boolalpha << clip << separator;
    result << "trgDev="     << target_device;
    return result.str();
}

void PriorBoxClusteredLayerTest::SetUp() {
    std::vector<InputShape> shapes;
    ov::element::Type model_type;
    std::vector<float> widths;
    std::vector<float> heights;
    std::vector<float> variances;
    float step_width;
    float step_height;
    float step;
    float offset;
    bool clip;
    priorBoxClusteredSpecificParams specParams;
    std::tie(specParams, model_type, shapes, targetDevice) = GetParam();
    std::tie(widths, heights, clip, step_width, step_height, step, offset, variances) = specParams;
    init_input_shapes(shapes);

    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[0]),
                               std::make_shared<ov::op::v0::Parameter>(model_type, inputDynamicShapes[1])};

    ov::op::v0::PriorBoxClustered::Attributes attributes;
    attributes.widths = widths;
    attributes.heights = heights;
    attributes.clip = clip;
    attributes.step_widths = step_width;
    attributes.step_heights = step_height;
    attributes.step = step;
    attributes.offset = offset;
    attributes.variances = variances;

    auto shape_of_1 = std::make_shared<ov::op::v3::ShapeOf>(params[0]);
    auto shape_of_2 = std::make_shared<ov::op::v3::ShapeOf>(params[1]);
    auto prior_box_clustered = std::make_shared<ov::op::v0::PriorBoxClustered>(shape_of_1, shape_of_2, attributes);

    auto result = std::make_shared<ov::op::v0::Result>(prior_box_clustered);
    function = std::make_shared<ov::Model>(result, params, "PB_Clustered");
}
}  // namespace test
}  // namespace ov
