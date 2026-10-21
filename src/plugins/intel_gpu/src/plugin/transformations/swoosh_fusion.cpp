// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swoosh_fusion.hpp"

#include <algorithm>
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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
namespace {

template <typename BinaryOp>
bool split_scalar_constant(const std::shared_ptr<ov::Node>& node, ov::Output<ov::Node>& data, float& value) {
    auto binary = ov::as_type_ptr<BinaryOp>(node);
    if (!binary) {
        return false;
    }

    for (size_t constant_index = 0; constant_index < 2; ++constant_index) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(binary->get_input_node_shared_ptr(constant_index));
        if (constant && ov::shape_size(constant->get_shape()) > 0 &&
            ov::op::util::get_single_value(constant, value, false)) {
            data = binary->input_value(1 - constant_index);
            return true;
        }
    }
    return false;
}

bool has_only_users(const ov::Output<ov::Node>& output, const std::vector<std::shared_ptr<ov::Node>>& expected) {
    const auto& targets = output.get_target_inputs();
    if (targets.size() != expected.size()) {
        return false;
    }
    return std::all_of(expected.begin(), expected.end(), [&](const std::shared_ptr<ov::Node>& node) {
        return std::any_of(targets.begin(), targets.end(), [&](const ov::Input<ov::Node>& target) {
            return target.get_node() == node.get();
        });
    });
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

    auto root_pattern = wrap_type<ov::op::v1::Add>({any_input(), any_input()});
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& matcher) {
        const auto root = matcher.get_match_root();
        if (transformation_callback(root)) {
            return false;
        }

        ov::Output<ov::Node> body_output;
        float final_bias = 0.0f;
        if (!split_scalar_constant<ov::op::v1::Add>(root, body_output, final_bias)) {
            return false;
        }

        auto body = body_output.get_node_shared_ptr();
        std::shared_ptr<ov::op::v1::Select> select;
        std::shared_ptr<ov::Node> scaled;
        float expected_slope = 0.0f;
        if (auto subtract = ov::as_type_ptr<ov::op::v1::Subtract>(body)) {
            select = ov::as_type_ptr<ov::op::v1::Select>(subtract->get_input_node_shared_ptr(0));
            scaled = subtract->get_input_node_shared_ptr(1);
            expected_slope = 0.08f;
        } else if (auto add = ov::as_type_ptr<ov::op::v1::Add>(body)) {
            for (size_t select_index = 0; select_index < 2; ++select_index) {
                select = ov::as_type_ptr<ov::op::v1::Select>(add->get_input_node_shared_ptr(select_index));
                if (select) {
                    scaled = add->get_input_node_shared_ptr(1 - select_index);
                    expected_slope = -0.08f;
                    break;
                }
            }
        }
        if (!select || !scaled) {
            return false;
        }

        ov::Output<ov::Node> direct_input;
        float slope = 0.0f;
        if (!split_scalar_constant<ov::op::v1::Multiply>(scaled, direct_input, slope)) {
            return false;
        }

        const auto data_type = direct_input.get_element_type();
        if (!matches_constant(slope, expected_slope, data_type)) {
            return false;
        }

        auto equal = ov::as_type_ptr<ov::op::v1::Equal>(select->get_input_node_shared_ptr(0));
        auto softplus = ov::as_type_ptr<ov::op::v4::SoftPlus>(select->get_input_node_shared_ptr(2));
        auto shifted = select->get_input_node_shared_ptr(1);
        if (!equal || !softplus || softplus->get_input_node_shared_ptr(0) != shifted) {
            return false;
        }

        ov::Output<ov::Node> equal_data;
        float equal_constant = 0.0f;
        const bool has_equal_constant = split_scalar_constant<ov::op::v1::Equal>(equal, equal_data, equal_constant);
        const bool is_positive_limit = (std::isinf(equal_constant) && equal_constant > 0.0f) ||
                                       (data_type == ov::element::f16 &&
                                        equal_constant == static_cast<float>(std::numeric_limits<ov::float16>::max()));
        if (!has_equal_constant || !is_positive_limit || equal_data.get_node_shared_ptr() != softplus) {
            return false;
        }

        ov::Output<ov::Node> shifted_input;
        float shift = 0.0f;
        if (!split_scalar_constant<ov::op::v1::Add>(shifted, shifted_input, shift) || shifted_input != direct_input) {
            return false;
        }

        const bool is_swoosh_l = matches_constant(shift, -4.0f, data_type) &&
                                 matches_constant(final_bias, -0.035f, data_type);
        const bool is_swoosh_r = matches_constant(shift, -1.0f, data_type) &&
                                 matches_constant(final_bias, -0.3132616877555847f, data_type);
        if (!is_swoosh_l && !is_swoosh_r) {
            return false;
        }

        if (root->get_output_partial_shape(0) != direct_input.get_partial_shape() ||
            root->get_output_element_type(0) != data_type) {
            return false;
        }

        if (!has_only_users(body->output(0), {root}) ||
            !has_only_users(scaled->output(0), {body}) ||
            !has_only_users(select->output(0), {body}) ||
            !has_only_users(equal->output(0), {select}) ||
            !has_only_users(softplus->output(0), {equal, select}) ||
            !has_only_users(shifted->output(0), {softplus, select})) {
            return false;
        }

        const float offset = is_swoosh_l ? 4.0f : 1.0f;
        const float bias = is_swoosh_l ? 0.035f : 0.3132616877555847f;
        auto swoosh = std::make_shared<ov::intel_gpu::op::Swoosh>(direct_input, offset, bias);
        swoosh->set_friendly_name(root->get_friendly_name());
        ov::copy_runtime_info({root, body, scaled, select, equal, softplus, shifted}, swoosh);
        ov::replace_node(root, swoosh);
        return true;
    };

    auto matcher = std::make_shared<Matcher>(root_pattern, "SwooshFusion");
    register_matcher(matcher, callback);
}

}  // namespace ov::intel_gpu
