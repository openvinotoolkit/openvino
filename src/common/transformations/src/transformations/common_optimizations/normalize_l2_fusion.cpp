// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/normalize_l2_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::NormalizeL2Fusion::NormalizeL2Fusion() {
    MATCHER_SCOPE(NormalizeL2Fusion);
    auto input = pass::pattern::any_input();

    auto exp = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto pow = std::make_shared<ov::op::v1::Power>(input, exp);
    auto axes = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(pow, axes);

    auto eps_const = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto max = std::make_shared<ov::op::v1::Maximum>(reduce_sum, eps_const);
    auto add = std::make_shared<ov::op::v1::Add>(reduce_sum, eps_const);
    auto max_or_add = std::make_shared<pattern::op::Or>(OutputVector{max, add});

    // Sqrt can be represented by Sqrt node or as Power node with exponent 0.5
    auto sqrt = std::make_shared<ov::op::v0::Sqrt>(max_or_add);
    auto exp2 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto pow_as_sqrt = std::make_shared<ov::op::v1::Power>(max_or_add, exp2);
    auto power_or_sqrt = std::make_shared<pattern::op::Or>(OutputVector{sqrt, pow_as_sqrt});

    // divide(input,sqrt(..)) can be represented as mul(input, power(..., -0.5f))
    auto divide = std::make_shared<ov::op::v1::Divide>(input, power_or_sqrt);
    auto exp3 = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto reversed_pow_as_sqrt = std::make_shared<ov::op::v1::Power>(max_or_add, exp3);
    auto mul = std::make_shared<ov::op::v1::Multiply>(input, reversed_pow_as_sqrt);
    auto divide_or_mul = std::make_shared<pattern::op::Or>(OutputVector{divide, mul});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto data_input = pattern_to_output.at(input);
        const auto exp_input = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(exp).get_node_shared_ptr());
        const auto axes_input = ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(axes).get_node_shared_ptr());
        const auto eps_attr =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(eps_const).get_node_shared_ptr());
        const auto exp2_input =
            pattern_to_output.count(exp2)
                ? ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(exp2).get_node_shared_ptr())
                : nullptr;
        const auto exp3_input =
            pattern_to_output.count(exp3)
                ? ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(exp3).get_node_shared_ptr())
                : nullptr;

        if (exp_input && !op::util::has_constant_value<float>(exp_input, 2.0f)) {
            return false;
        }

        if (exp2_input && !op::util::has_constant_value<float>(exp2_input, 0.5f)) {
            return false;
        }

        if (exp3_input && !op::util::has_constant_value<float>(exp3_input, -0.5f)) {
            return false;
        }

        if (!eps_attr || shape_size(eps_attr->get_shape()) > 1) {
            return false;
        }

        const auto eps_attr_value = eps_attr->cast_vector<float>()[0];
        op::EpsMode mode;
        Output<Node> eps_node;
        if (pattern_to_output.count(max)) {
            mode = op::EpsMode::MAX;
            eps_node = pattern_to_output.at(max);
        } else if (pattern_to_output.count(add)) {
            mode = op::EpsMode::ADD;
            eps_node = pattern_to_output.at(add);
        } else {
            return false;
        }

        auto normalize_l2 = std::make_shared<ov::op::v0::NormalizeL2>(data_input, axes_input, eps_attr_value, mode);
        if (transformation_callback(normalize_l2)) {
            return false;
        }
        normalize_l2->set_friendly_name(m.get_match_root()->get_friendly_name());

        OutputVector outputs_to_replace{pattern_to_output.at(pow), pattern_to_output.at(reduce_sum), eps_node};
        if (pattern_to_output.count(mul)) {
            outputs_to_replace.emplace_back(pattern_to_output.at(mul));
        }
        if (pattern_to_output.count(divide)) {
            outputs_to_replace.emplace_back(pattern_to_output.at(divide));
        }
        if (pattern_to_output.count(sqrt)) {
            outputs_to_replace.emplace_back(pattern_to_output.at(sqrt));
        }
        if (pattern_to_output.count(pow_as_sqrt)) {
            outputs_to_replace.emplace_back(pattern_to_output.at(pow_as_sqrt));
        }
        if (pattern_to_output.count(reversed_pow_as_sqrt)) {
            outputs_to_replace.emplace_back(pattern_to_output.at(reversed_pow_as_sqrt));
        }
        if (pattern_to_output.count(max)) {
            outputs_to_replace.emplace_back(pattern_to_output.at(max));
        }
        if (pattern_to_output.count(add)) {
            outputs_to_replace.emplace_back(pattern_to_output.at(add));
        }

        ov::copy_runtime_info(as_node_vector(outputs_to_replace), normalize_l2);
        ov::replace_node(m.get_match_root(), normalize_l2);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(divide_or_mul, matcher_name);
    register_matcher(m, callback);
}
