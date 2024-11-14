// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/hsigmoid_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::HSigmoidFusionWithReluDiv::HSigmoidFusionWithReluDiv() {
    MATCHER_SCOPE(HSigmoidFusionWithReluDiv);
    // Replaces a sub-graph ((min(Relu(x + 3), 6)) / 6 with a HSigmoid op.
    auto input = pass::pattern::any_input();
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, add_constant});
    auto relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({add});
    auto min_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto min = ov::pass::pattern::wrap_type<ov::op::v1::Minimum>({relu, min_constant});
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({min, div_constant});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto min_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto div_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value<float>(add_const_value, 3.0) &&
                                     op::util::has_constant_value<float>(min_const_value, 6.0) &&
                                     op::util::has_constant_value<float>(div_const_value, 6.0);

        if (!valid_constant_values) {
            return false;
        }

        auto hsigmoid = register_new_node<ov::op::v5::HSigmoid>(x_output);

        hsigmoid->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(relu).get_node_shared_ptr(),
                pattern_to_output.at(min).get_node_shared_ptr(),
                pattern_to_output.at(div).get_node_shared_ptr(),
            },
            hsigmoid);
        ov::replace_node(m.get_match_root(), hsigmoid);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(div, matcher_name);
    register_matcher(m, callback);
}

ov::pass::HSigmoidFusionWithReluMul::HSigmoidFusionWithReluMul() {
    MATCHER_SCOPE(HSigmoidFusionWithReluMul);
    // Replaces a sub-graph ((min(Relu(x + 3), 6)) * const(1/6) with a HSigmoid op.
    auto input = pass::pattern::any_input();
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, add_constant});
    auto relu = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({add});
    auto min_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto min = ov::pass::pattern::wrap_type<ov::op::v1::Minimum>({relu, min_constant});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_second = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({min, mul_constant});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto min_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto mul_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value<float>(add_const_value, 3.0f) &&
                                     op::util::has_constant_value<float>(min_const_value, 6.0f) &&
                                     op::util::has_constant_value<float>(mul_const_value, (1.0f / 6.0f), 0.0001f);

        if (!valid_constant_values) {
            return false;
        }

        auto hsigmoid = register_new_node<ov::op::v5::HSigmoid>(x_output);

        hsigmoid->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(relu).get_node_shared_ptr(),
                               pattern_to_output.at(min).get_node_shared_ptr(),
                               pattern_to_output.at(mul_second).get_node_shared_ptr()},
                              hsigmoid);
        ov::replace_node(m.get_match_root(), hsigmoid);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_second, matcher_name);
    register_matcher(m, callback);
}

ov::pass::HSigmoidFusionWithoutRelu::HSigmoidFusionWithoutRelu() {
    MATCHER_SCOPE(HSigmoidFusionWithoutRelu);
    // Replaces a sub-graph (min(max(x + 3, 0), 6) / 6) with a HSigmoid op.
    auto input = pass::pattern::any_input();
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, add_constant});
    auto max_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto max = ov::pass::pattern::wrap_type<ov::op::v1::Maximum>({add, max_constant});
    auto min_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto min = ov::pass::pattern::wrap_type<ov::op::v1::Minimum>({max, min_constant});
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({min, div_constant});
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({input, div});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto max_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(max_constant).get_node_shared_ptr());
        auto min_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(min_constant).get_node_shared_ptr());
        auto div_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value<float>(add_const_value, 3.0f) &&
                                     op::util::has_constant_value<float>(max_const_value, 0.0f) &&
                                     op::util::has_constant_value<float>(min_const_value, 6.0f) &&
                                     op::util::has_constant_value<float>(div_const_value, 6.0f);

        if (!valid_constant_values) {
            return false;
        }

        auto hsigmoid = register_new_node<ov::op::v5::HSigmoid>(x_output);

        hsigmoid->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(max).get_node_shared_ptr(),
                               pattern_to_output.at(min).get_node_shared_ptr(),
                               pattern_to_output.at(div).get_node_shared_ptr()},
                              hsigmoid);
        ov::replace_node(m.get_match_root(), hsigmoid);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(div, matcher_name);
    register_matcher(m, callback);
}

ov::pass::HSigmoidFusionWithClampMul::HSigmoidFusionWithClampMul() {
    MATCHER_SCOPE(HSigmoidFusionWithClampMul);
    // Replaces a sub-graph (Clamp(x + 3, 0, 6) * const(1/6)) with a HSigmoid op.
    auto input = pass::pattern::any_input();
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, add_constant});
    auto clamp = ov::pass::pattern::wrap_type<ov::op::v0::Clamp>({add});
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_first = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({clamp, mul_constant});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto mul_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(mul_constant).get_node_shared_ptr());

        bool valid_constant_values = op::util::has_constant_value(add_const_value, 3.0) &&
                                     op::util::has_constant_value(mul_const_value, (1.0 / 6.0), 0.0001);

        if (!valid_constant_values) {
            return false;
        }

        auto clamp_node = ov::as_type_ptr<ov::op::v0::Clamp>(pattern_to_output.at(clamp).get_node_shared_ptr());
        if (!clamp_node || clamp_node->get_min() != 0 || clamp_node->get_max() != 6)
            return false;

        auto hsigmoid = register_new_node<ov::op::v5::HSigmoid>(x_output);

        hsigmoid->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(clamp).get_node_shared_ptr(),
                               pattern_to_output.at(mul_first).get_node_shared_ptr()},
                              hsigmoid);
        ov::replace_node(m.get_match_root(), hsigmoid);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_first, matcher_name);
    register_matcher(m, callback);
}

ov::pass::HSigmoidFusionWithClampDiv::HSigmoidFusionWithClampDiv() {
    MATCHER_SCOPE(HSigmoidFusionWithClampDiv);
    // Replaces a sub-graph (Clamp(x + 3, 0, 6) / 6) with a HSigmoid op.
    auto input = pass::pattern::any_input();
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, add_constant});
    auto clamp = ov::pass::pattern::wrap_type<ov::op::v0::Clamp>({add});
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto div = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({clamp, div_constant});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto x_output = pattern_to_output.at(input);

        auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        auto div_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(div_constant).get_node_shared_ptr());

        bool valid_constant_values =
            op::util::has_constant_value(add_const_value, 3.0) && op::util::has_constant_value(div_const_value, 6.0);

        if (!valid_constant_values) {
            return false;
        }

        auto clamp_node = ov::as_type_ptr<ov::op::v0::Clamp>(pattern_to_output.at(clamp).get_node_shared_ptr());
        if (!clamp_node || clamp_node->get_min() != 0 || clamp_node->get_max() != 6)
            return false;

        auto hsigmoid = register_new_node<ov::op::v5::HSigmoid>(x_output);

        hsigmoid->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(clamp).get_node_shared_ptr(),
                               pattern_to_output.at(div).get_node_shared_ptr()},
                              hsigmoid);
        ov::replace_node(m.get_match_root(), hsigmoid);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(div, matcher_name);
    register_matcher(m, callback);
}
