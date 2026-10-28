// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swoosh_fusion.hpp"

#include <cmath>
#include <limits>

#include "intel_gpu/op/swoosh.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

namespace swoosh_constants = op::swoosh_constants;

bool get_scalar_constant(const ov::Output<ov::Node>& output, float& value) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    return constant && ov::shape_size(constant->get_shape()) > 0 &&
           ov::op::util::get_single_value(constant, value, false);
}

bool matches_constant(float value, float expected, const ov::element::Type& data_type) {
    if (data_type == ov::element::f32) {
        return value == expected;
    }
    if (data_type == ov::element::f16) {
        return value == static_cast<float>(ov::float16(expected));
    }
    return false;
}

}  // namespace

SwooshFusion::SwooshFusion() {
    using namespace ov::pass::pattern;

    auto input_m = any_input();
    auto shift_constant_m = wrap_type<ov::op::v0::Constant>();
    auto shifted_lhs_m = wrap_type<ov::op::v1::Add>({input_m, shift_constant_m}, consumers_count(2));
    auto shifted_rhs_m = wrap_type<ov::op::v1::Add>({shift_constant_m, input_m}, consumers_count(2));
    auto shifted_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{shifted_lhs_m, shifted_rhs_m});
    auto softplus_m = wrap_type<ov::op::v4::SoftPlus>({shifted_m}, consumers_count(2));

    auto limit_constant_m = wrap_type<ov::op::v0::Constant>();
    auto equal_lhs_m = wrap_type<ov::op::v1::Equal>({softplus_m, limit_constant_m}, consumers_count(1));
    auto equal_rhs_m = wrap_type<ov::op::v1::Equal>({limit_constant_m, softplus_m}, consumers_count(1));
    auto equal_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{equal_lhs_m, equal_rhs_m});
    auto select_m = wrap_type<ov::op::v1::Select>({equal_m, shifted_m, softplus_m}, consumers_count(1));

    auto slope_constant_m = wrap_type<ov::op::v0::Constant>();
    auto scaled_lhs_m = wrap_type<ov::op::v1::Multiply>({input_m, slope_constant_m}, consumers_count(1));
    auto scaled_rhs_m = wrap_type<ov::op::v1::Multiply>({slope_constant_m, input_m}, consumers_count(1));
    auto scaled_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{scaled_lhs_m, scaled_rhs_m});

    auto subtract_m = wrap_type<ov::op::v1::Subtract>({select_m, scaled_m}, consumers_count(1));
    auto body_lhs_m = wrap_type<ov::op::v1::Add>({select_m, scaled_m}, consumers_count(1));
    auto body_rhs_m = wrap_type<ov::op::v1::Add>({scaled_m, select_m}, consumers_count(1));
    auto body_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{subtract_m, body_lhs_m, body_rhs_m});

    auto bias_constant_m = wrap_type<ov::op::v0::Constant>();
    auto root_lhs_m = wrap_type<ov::op::v1::Add>({body_m, bias_constant_m});
    auto root_rhs_m = wrap_type<ov::op::v1::Add>({bias_constant_m, body_m});
    auto root_m = std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{root_lhs_m, root_rhs_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& matcher) {
        const auto& pattern_map = matcher.get_pattern_value_map();
        const auto root = pattern_map.at(root_m).get_node_shared_ptr();
        if (transformation_callback(root)) {
            return false;
        }

        const auto direct_input = pattern_map.at(input_m);
        const auto body = pattern_map.at(body_m).get_node_shared_ptr();
        const auto scaled = pattern_map.at(scaled_m).get_node_shared_ptr();
        const auto select = pattern_map.at(select_m).get_node_shared_ptr();
        const auto equal = pattern_map.at(equal_m).get_node_shared_ptr();
        const auto softplus = pattern_map.at(softplus_m).get_node_shared_ptr();
        const auto shifted = pattern_map.at(shifted_m).get_node_shared_ptr();
        const auto data_type = direct_input.get_element_type();

        float shift = 0.0f;
        float slope = 0.0f;
        float equal_constant = 0.0f;
        float final_bias = 0.0f;
        if (!get_scalar_constant(pattern_map.at(shift_constant_m), shift) ||
            !get_scalar_constant(pattern_map.at(slope_constant_m), slope) ||
            !get_scalar_constant(pattern_map.at(limit_constant_m), equal_constant) ||
            !get_scalar_constant(pattern_map.at(bias_constant_m), final_bias)) {
            return false;
        }

        const float expected_slope = ov::is_type<ov::op::v1::Subtract>(body) ? swoosh_constants::slope : -swoosh_constants::slope;
        if (!matches_constant(slope, expected_slope, data_type)) {
            return false;
        }

        const bool is_positive_limit = (std::isinf(equal_constant) && equal_constant > 0.0f) ||
                                       (data_type == ov::element::f16 &&
                                        equal_constant == static_cast<float>(std::numeric_limits<ov::float16>::max()));
        if (!is_positive_limit) {
            return false;
        }

        const bool is_swoosh_l =
            matches_constant(shift, -swoosh_constants::swoosh_l_offset, data_type) && matches_constant(final_bias, -swoosh_constants::swoosh_l_bias, data_type);
        const bool is_swoosh_r =
            matches_constant(shift, -swoosh_constants::swoosh_r_offset, data_type) && matches_constant(final_bias, -swoosh_constants::swoosh_r_bias, data_type);
        if (!is_swoosh_l && !is_swoosh_r) {
            return false;
        }

        if (root->get_output_partial_shape(0) != direct_input.get_partial_shape() ||
            root->get_output_element_type(0) != data_type) {
            return false;
        }

        const float offset = is_swoosh_l ? swoosh_constants::swoosh_l_offset : swoosh_constants::swoosh_r_offset;
        const float bias = is_swoosh_l ? swoosh_constants::swoosh_l_bias : swoosh_constants::swoosh_r_bias;
        auto swoosh = std::make_shared<ov::intel_gpu::op::Swoosh>(direct_input, offset, bias);
        swoosh->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info({root, body, scaled, select, equal, softplus, shifted}, swoosh);
        ov::replace_node(root, swoosh);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(root_m, "SwooshFusion");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
