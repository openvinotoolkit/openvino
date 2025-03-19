// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/lin_op_sequence_fusion.hpp"

#include <memory>
#include <vector>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace {
const auto is_eltwise_supported_type = [](const Output<Node>& output) -> bool {
    const auto is_single_output = pass::pattern::consumers_count(1);
    return is_single_output(output) && output.get_node()->has_evaluate();
};
}

ov::pass::AddMultiplyFusion::AddMultiplyFusion() {
    MATCHER_SCOPE(AddMultiplyFusion);
    // Create Add->Multiply pattern where Add has exactly one consumer
    auto m_data = pass::pattern::any_input();
    auto m_add_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_add = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_data, m_add_constant}, pattern::consumers_count(1));
    auto m_mul_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_add, m_mul_constant});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) -> bool {
        auto& label_to_output = m.get_pattern_value_map();

        auto mul = label_to_output[m_mul].get_node_shared_ptr();
        auto add = label_to_output[m_add].get_node_shared_ptr();

        if (transformation_callback(mul)) {
            return false;
        }

        Output<Node> input = label_to_output[m_data];
        Output<Node> mul_const = label_to_output[m_mul_constant];
        Output<Node> add_const = label_to_output[m_add_constant];

        if ((input.get_element_type() != mul_const.get_element_type()) ||
            (add_const.get_element_type() != mul_const.get_element_type())) {
            return false;
        }

        // Replace Add->Multiply with Multiply->Add
        // As new Multiply can be fused with operation above it we add this Multiply
        // to the list of operations that will be used in additional matching.
        auto new_mul = register_new_node<ov::op::v1::Multiply>(input, mul_const);

        // Add two constants using opset3::Add constant folding and create new Add operation
        auto new_const = op::util::make_try_fold<ov::op::v1::Multiply>(add_const, mul_const);
        auto new_add = std::make_shared<ov::op::v1::Add>(new_mul, new_const);

        copy_runtime_info({add, mul}, {new_mul, new_add, new_const});
        new_add->set_friendly_name(mul->get_friendly_name());
        replace_node(mul, new_add);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_mul, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::AddAddFusion::AddAddFusion() {
    MATCHER_SCOPE(AddAddFusion);
    // Create Add->Add pattern where first Add has exactly one consumer
    auto m_data = pass::pattern::any_input();
    auto m_add1_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_add1 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_data, m_add1_constant}, pattern::consumers_count(1));
    auto m_add2_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_add2 = ov::pass::pattern::wrap_type<ov::op::v1::Add>({m_add1, m_add2_constant});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) -> bool {
        auto& label_to_output = m.get_pattern_value_map();

        auto add1 = label_to_output[m_add1].get_node_shared_ptr();
        auto add2 = label_to_output[m_add2].get_node_shared_ptr();

        Output<Node> input = label_to_output[m_data];
        Output<Node> add1_const = label_to_output[m_add1_constant];
        Output<Node> add2_const = label_to_output[m_add2_constant];

        // Replace Add->Add with single Add
        // Add operation will be added to the list of ops requested for pattern matching
        auto new_const = op::util::make_try_fold<ov::op::v1::Add>(add1_const, add2_const);
        auto new_add = register_new_node<ov::op::v1::Add>(input, new_const);

        copy_runtime_info({add1, add2}, {new_add, new_const});
        new_add->set_friendly_name(add2->get_friendly_name());
        replace_node(add2, new_add);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_add2, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::MultiplyMultiplyFusion::MultiplyMultiplyFusion() {
    MATCHER_SCOPE(MultiplyMultiplyFusion);
    // Create Multiply->Multiply pattern where first Multiply has exactly one consumer
    auto m_data = pass::pattern::any_input();
    auto m_mul1_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_mul1 =
        ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_data, m_mul1_constant}, is_eltwise_supported_type);
    auto m_mul2_constant = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto m_mul2 = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({m_mul1, m_mul2_constant});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) -> bool {
        auto& label_to_output = m.get_pattern_value_map();

        auto mul1 = label_to_output[m_mul1].get_node_shared_ptr();
        auto mul2 = label_to_output[m_mul2].get_node_shared_ptr();

        Output<Node> input = label_to_output[m_data];
        Output<Node> mul1_const = label_to_output[m_mul1_constant];
        Output<Node> mul2_const = label_to_output[m_mul2_constant];

        // Replace Multiply->Multiply with single Multiply
        // Multiply operation will be added to the list of ops requested for pattern matching
        auto new_const = op::util::make_try_fold<ov::op::v1::Multiply>(mul1_const, mul2_const);
        auto new_mul = register_new_node<ov::op::v1::Multiply>(input, new_const);

        copy_runtime_info({mul1, mul2}, {new_mul, new_const});
        new_mul->set_friendly_name(mul2->get_friendly_name());
        replace_node(mul2, new_mul);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(m_mul2, matcher_name);
    this->register_matcher(m, callback);
}
