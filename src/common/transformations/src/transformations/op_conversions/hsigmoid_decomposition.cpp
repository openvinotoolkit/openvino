// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/hsigmoid_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/hsigmoid.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::HSigmoidDecomposition::HSigmoidDecomposition() {
    MATCHER_SCOPE(HSigmoidDecomposition);
    // Decomposes HSigmoid(x) op into sub-graph (min(Relu(x + 3), 6) * const(1/6)
    auto hsigmoid = ov::pass::pattern::wrap_type<ov::op::v5::HSigmoid>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto hsigmoid_node = pattern_to_output.at(hsigmoid).get_node_shared_ptr();

        if (transformation_callback(hsigmoid_node)) {
            return false;
        }

        auto input_type = hsigmoid_node->input_value(0).get_element_type();
        auto add_constant = ov::op::v0::Constant::create(input_type, ov::Shape{}, {3.0});
        auto add = std::make_shared<ov::op::v1::Add>(hsigmoid_node->input_value(0), add_constant);
        auto relu = std::make_shared<ov::op::v0::Relu>(add);
        auto min_constant = ov::op::v0::Constant::create(input_type, ov::Shape{}, {6.0});
        auto min = register_new_node<ov::op::v1::Minimum>(relu, min_constant);
        auto mul_constant = ov::op::v0::Constant::create(input_type, ov::Shape{}, {(1.0 / 6.0)});  // const(1/6)
        auto mul = std::make_shared<ov::op::v1::Multiply>(min, mul_constant);

        mul->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::copy_runtime_info(hsigmoid_node, {add_constant, add, relu, min_constant, min, min_constant, mul});
        ov::replace_node(m.get_match_root(), mul);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(hsigmoid, matcher_name);
    register_matcher(m, callback);
}
