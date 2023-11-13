// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "move_eltwise_up_data_movement.hpp"

#include <memory>
#include <vector>
#include <numeric>

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "itt.hpp"


namespace {
    bool is_data_movement_operation(const std::shared_ptr<ov::Node>& node) {
        return ov::is_type<ngraph::op::v0::Squeeze>(node) ||
               ov::is_type<ngraph::op::v0::Unsqueeze>(node) ||
               ov::is_type<ngraph::op::v1::Reshape>(node) ||
               ov::is_type<ngraph::op::v1::Transpose>(node) ||
               ov::is_type<ngraph::op::v0::ShuffleChannels>(node) ||
               ov::is_type<ngraph::op::v7::Roll>(node) ||
               ov::is_type<ngraph::op::v0::ReverseSequence>(node) ||
               ov::is_type<ngraph::op::v0::DepthToSpace>(node) ||
               ov::is_type<ngraph::op::v1::BatchToSpace>(node) ||
               ov::is_type<ngraph::op::v1::Broadcast>(node) ||
               ov::is_type<ngraph::op::v3::Broadcast>(node) ||
               ov::is_type<ngraph::op::v1::Gather>(node) ||
               ov::is_type<ngraph::op::v7::Gather>(node) ||
               ov::is_type<ngraph::op::v8::Gather>(node);
    }

    bool is_scalar_like(const std::shared_ptr<ov::Node>& node) {
        auto constantNode = std::dynamic_pointer_cast<ngraph::opset8::Constant>(node);
        return constantNode != nullptr && shape_size(constantNode->get_shape()) == 1;
    }
} // namespace

ov::intel_cpu::MoveEltwiseUpThroughDataMov::MoveEltwiseUpThroughDataMov() {
    MATCHER_SCOPE(MoveEltwiseUpThroughDataMov);
    auto eltwise_pattern = ov::pass::pattern::wrap_type<ngraph::op::util::UnaryElementwiseArithmetic,
                                                      ngraph::op::util::BinaryElementwiseArithmetic>(ov::pass::pattern::has_static_rank());

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto eltwise = pattern_map.at(eltwise_pattern).get_node_shared_ptr();
        if (transformation_callback(eltwise)) {
            return false;
        }

        if (eltwise->get_output_size() == 0 ||
            eltwise->get_input_size() == 0 ||
            eltwise->get_output_element_type(0) != eltwise->get_input_element_type(0) ||
            eltwise->get_output_target_inputs(0).size() != 1) {
            return false;
        }

        bool is_binary_op = std::dynamic_pointer_cast<ngraph::op::util::BinaryElementwiseArithmetic>(eltwise) != nullptr;
        if (is_binary_op && !is_scalar_like(eltwise->get_input_node_shared_ptr(1))) {
            return false;
        }



        auto current = eltwise->get_input_node_shared_ptr(0);
        auto child = eltwise;

        while (is_data_movement_operation(current)) {
            if (current->get_output_size() != 1 ||
                current->get_output_target_inputs(0).size() != 1 ||
                current->get_output_element_type(0) != current->get_input_element_type(0)) {
                return false;
            }

            child = current;
            current = current->get_input_node_shared_ptr(0);
        }

        // now current is the first not data movement op
        if (child == eltwise) {
            return false;
        }

        // eltwise constant shape should match new input shape
        if (is_binary_op && current->get_output_partial_shape(0).rank().get_length() != eltwise->get_input_partial_shape(1).rank().get_length()) {
            auto old_eltwise_const = std::dynamic_pointer_cast<ngraph::opset8::Constant>(eltwise->get_input_node_shared_ptr(1));
            auto new_constant = std::make_shared<ngraph::opset8::Constant>(*old_eltwise_const.get(), ov::Shape{});
            ov::copy_runtime_info(old_eltwise_const, new_constant);
            ov::replace_node(old_eltwise_const, new_constant);
        }
        ov::replace_output_update_name(eltwise->output(0), eltwise->input_value(0));

        ov::OutputVector eltwiseInputs = eltwise->input_values();
        eltwiseInputs[0] = child->input_value(0);
        auto newEltwise = eltwise->clone_with_new_inputs(eltwiseInputs);
        // WA: it's necessary to set empty friendly name here
        // to avoid name duplication in TypeRelaxed cases
        newEltwise->set_friendly_name("");
        ov::copy_runtime_info(eltwise, newEltwise);

        ov::OutputVector childInputs = child->input_values();
        childInputs[0] = newEltwise;
        auto newChild = child->clone_with_new_inputs(childInputs);
        ov::copy_runtime_info(child, newChild);
        newChild->set_friendly_name(child->get_friendly_name());

        ov::replace_node(child, newChild);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(eltwise_pattern, matcher_name);
    register_matcher(m, callback);
}
