// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/hard_sigmoid_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hard_sigmoid.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::HardSigmoidDecomposition::HardSigmoidDecomposition() {
    MATCHER_SCOPE(HardSigmoidDecomposition);
    // Decomposes HardSigmoid(x) op into sub-graph max(0, min(1, alpha * x + beta))
    auto hard_sigmoid = ov::pass::pattern::wrap_type<ov::op::v0::HardSigmoid>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto hard_sigmoid_node = pattern_to_output.at(hard_sigmoid).get_node_shared_ptr();

        if (transformation_callback(hard_sigmoid_node)) {
            return false;
        }

        auto alpha_constant = hard_sigmoid_node->get_input_node_shared_ptr(1);
        auto multiply = std::make_shared<ov::op::v1::Multiply>(hard_sigmoid_node->input_value(0), alpha_constant);

        auto betta_constant = hard_sigmoid_node->get_input_node_shared_ptr(2);
        auto add = std::make_shared<ov::op::v1::Add>(multiply, betta_constant);

        auto input_type = hard_sigmoid_node->input_value(0).get_element_type();
        auto min_constant = ov::op::v0::Constant::create(input_type, ov::Shape{}, {1.f});
        auto min = std::make_shared<ov::op::v1::Minimum>(add, min_constant);

        auto max_constant = ov::op::v0::Constant::create(input_type, ov::Shape{}, {0.0});
        auto max = std::make_shared<ov::op::v1::Maximum>(min, max_constant);

        max->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(hard_sigmoid_node,
                              {alpha_constant, multiply, betta_constant, add, min_constant, min, max_constant, max});
        ov::replace_node(m.get_match_root(), max);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(hard_sigmoid, matcher_name);
    register_matcher(m, callback);
}
