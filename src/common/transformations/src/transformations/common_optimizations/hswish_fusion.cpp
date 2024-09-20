// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/hswish_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/hswish.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::HSwishFusionWithReluDiv::HSwishFusionWithReluDiv() {
    MATCHER_SCOPE(HSwishFusionWithReluDiv);
    // Replaces a sub-graph (x * (min(Relu(x + 3), 6)) / 6 with a HSwish op.
    auto input = pass::pattern::any_input();
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = std::make_shared<ov::op::v1::Add>(input, add_constant);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto min_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto min = std::make_shared<ov::op::v1::Minimum>(relu, min_constant);
    auto mul = std::make_shared<ov::op::v1::Multiply>(input, min);
    auto div_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto div = std::make_shared<ov::op::v1::Divide>(mul, div_constant);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
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

        auto hswish = std::make_shared<ov::op::v4::HSwish>(x_output);

        hswish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(
            {
                pattern_to_output.at(add_constant).get_node_shared_ptr(),
                pattern_to_output.at(add).get_node_shared_ptr(),
                pattern_to_output.at(relu).get_node_shared_ptr(),
                pattern_to_output.at(min_constant).get_node_shared_ptr(),
                pattern_to_output.at(min).get_node_shared_ptr(),
                pattern_to_output.at(mul).get_node_shared_ptr(),
                pattern_to_output.at(div_constant).get_node_shared_ptr(),
                pattern_to_output.at(div).get_node_shared_ptr(),
            },
            hswish);
        ov::replace_node(m.get_match_root(), hswish);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(div, matcher_name);
    register_matcher(m, callback);
}

ov::pass::HSwishFusionWithReluMul::HSwishFusionWithReluMul() {
    MATCHER_SCOPE(HSwishFusionWithReluMul);
    // Replaces a sub-graph (x * (min(Relu(x + 3), 6)) * const(1/6) with a HSwish op.
    auto input = pass::pattern::any_input();
    auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto add = std::make_shared<ov::op::v1::Add>(input, add_constant);
    auto relu = std::make_shared<ov::op::v0::Relu>(add);
    auto min_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto min = std::make_shared<ov::op::v1::Minimum>(relu, min_constant);
    auto mul_first = std::make_shared<ov::op::v1::Multiply>(input, min);
    auto mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto mul_second = std::make_shared<ov::op::v1::Multiply>(mul_first, mul_constant);

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
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

        auto hswish = std::make_shared<ov::op::v4::HSwish>(x_output);

        hswish->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(add_constant).get_node_shared_ptr(),
                               pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(relu).get_node_shared_ptr(),
                               pattern_to_output.at(min_constant).get_node_shared_ptr(),
                               pattern_to_output.at(min).get_node_shared_ptr(),
                               pattern_to_output.at(mul_first).get_node_shared_ptr(),
                               pattern_to_output.at(mul_constant).get_node_shared_ptr(),
                               pattern_to_output.at(mul_second).get_node_shared_ptr()},
                              hswish);
        ov::replace_node(m.get_match_root(), hswish);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_second, matcher_name);
    register_matcher(m, callback);
}

ov::pass::HSwishFusionWithHSigmoid::HSwishFusionWithHSigmoid() {
    MATCHER_SCOPE(HSwishFusionWithHSigmoid);
    // Replaces a sub-graph x * HSigmoid(x) with a HSwish op.
    auto input = pattern::any_input();
    auto hsigmoid_pattern = pattern::wrap_type<ov::op::v5::HSigmoid>({input}, pattern::consumers_count(1));
    auto mul_pattern = pattern::wrap_type<ov::op::v1::Multiply>({input, hsigmoid_pattern});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        auto hsigmoid = pattern_to_output.at(hsigmoid_pattern).get_node_shared_ptr();
        auto mul = pattern_to_output.at(mul_pattern).get_node_shared_ptr();

        auto hswish = std::make_shared<ov::op::v4::HSwish>(pattern_to_output.at(input));
        hswish->set_friendly_name(mul->get_friendly_name());
        ov::copy_runtime_info({hsigmoid, mul}, hswish);
        ov::replace_node(mul, hswish);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul_pattern, matcher_name);
    register_matcher(m, callback);
}

ov::pass::HSwishFusionWithClamp::HSwishFusionWithClamp() {
    MATCHER_SCOPE(HSwishFusionWithClampMul);
    // Replaces a sub-graph (Clamp(x + 3, 0, 6) * x) with a HSwish * 6.
    const auto input = pass::pattern::any_input();
    const auto add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    const auto add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({input, add_constant});
    const auto clamp = ov::pass::pattern::wrap_type<ov::op::v0::Clamp>({add});
    const auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({clamp, input});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_to_output = m.get_pattern_value_map();
        const auto x_output = pattern_to_output.at(input);
        const auto add_const_value =
            ov::as_type_ptr<ov::op::v0::Constant>(pattern_to_output.at(add_constant).get_node_shared_ptr());
        if (!op::util::has_constant_value(add_const_value, 3.0)) {
            return false;
        }

        const auto clamp_node = ov::as_type_ptr<ov::op::v0::Clamp>(pattern_to_output.at(clamp).get_node_shared_ptr());
        if (!clamp_node || clamp_node->get_min() != 0 || clamp_node->get_max() != 6)
            return false;

        auto hswish = std::make_shared<ov::op::v4::HSwish>(x_output);
        auto new_mul_const = std::make_shared<ov::op::v0::Constant>(add_const_value->get_element_type(),
                                                                    Shape{},
                                                                    std::vector<float>{6.0});
        auto new_mul = std::make_shared<ov::op::v1::Multiply>(hswish, new_mul_const);

        new_mul->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info({pattern_to_output.at(add).get_node_shared_ptr(),
                               pattern_to_output.at(clamp).get_node_shared_ptr(),
                               pattern_to_output.at(mul).get_node_shared_ptr()},
                              {hswish, new_mul_const, new_mul});
        ov::replace_node(m.get_match_root(), new_mul);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}
