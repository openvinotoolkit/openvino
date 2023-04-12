// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mark_dequantization_subgraph.hpp"

#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/rt_info/dequantization_node.hpp>
#include <transformations/rt_info/disable_constant_folding.hpp>

static bool is_constfoldable(const ov::Output<ov::Node>& output) {
    auto status = true;
    std::deque<ov::Node*> nodes_to_calculate = {output.get_node()};

    while (status && !nodes_to_calculate.empty()) {
        auto current_node = nodes_to_calculate.front();
        nodes_to_calculate.pop_front();

        if (current_node->get_input_size() == 0 && !ov::is_type<ov::op::v0::Constant>(current_node)) {
            status = false;
        } else {
            // not a leaf, not a shape_of -- continue to search
            for (const auto& input_value : current_node->input_values()) {
                const auto& input_node = input_value.get_node();
                nodes_to_calculate.push_front(input_node);
            }
        }
    }
    return status;
}

ov::pass::MarkDequantizationSubgraph::MarkDequantizationSubgraph(const element::TypeVector& precisions) {
    // Dequantization subgraph may have two forms: with and without Subtract
    //
    //    Input                                 Input
    //      |                                     |
    //   Convert  zero point           OR       Convert   scale
    //       \     /                               \      /
    //       Subtract   scale                      Multiply
    //           \      /
    //           Multiply
    //
    auto input_pattern = pattern::any_input();
    auto convert_pattern = pattern::wrap_type<opset10::Convert>({input_pattern}, pattern::consumers_count(1));
    auto zero_point_pattern = pattern::any_input();
    auto subtract_pattern = pattern::wrap_type<opset10::Subtract>({convert_pattern, zero_point_pattern});
    auto multiply_pattern = pattern::wrap_type<opset10::Multiply>({subtract_pattern, pattern::any_input()});
    auto multiply_no_subtract_pattern = pattern::wrap_type<opset10::Multiply>({convert_pattern, pattern::any_input()});
    auto root = std::make_shared<pattern::op::Or>(OutputVector{multiply_pattern, multiply_no_subtract_pattern});

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) -> bool {
        const auto& pattern_map = m.get_pattern_value_map();
        auto convert = pattern_map.at(convert_pattern).get_node_shared_ptr();
        auto input = pattern_map.at(input_pattern).get_node_shared_ptr();
        const auto multiply = m.get_match_root();

        auto subtract_it = pattern_map.find(subtract_pattern);
        if (subtract_it == pattern_map.end()) {
            for (size_t i = 0; i < multiply->get_input_size(); i++) {
                const auto node = ov::as_type_ptr<opset10::Convert>(multiply->get_input_node_shared_ptr(i));
                if (node && std::find(precisions.begin(), precisions.end(), node->get_input_element_type(0)) !=
                                precisions.end()) {
                    convert = node;
                    input = convert->get_input_node_shared_ptr(0);
                }
            }
        }

        const auto& input_precision = input->get_output_element_type(0);
        // validation by Convert operation input precisions
        if (std::find(precisions.begin(), precisions.end(), input_precision) == precisions.end()) {
            return false;
        }

        if (is_constfoldable(input)) {
            // disable ConstantFolding if dequantization subgraph is on constant data
            ov::disable_constant_folding(convert);
        }

        if (subtract_it != pattern_map.end()) {
            // mark Subtract as dequantization node
            ov::mark_as_dequantization_node(subtract_it->second.get_node_shared_ptr());
            auto zero_point = pattern_map.at(zero_point_pattern).get_node_shared_ptr();
            if (ov::is_type<opset10::Convert>(zero_point) && input_precision == zero_point->get_input_element_type(0) &&
                ov::is_type<opset10::Constant>(zero_point->get_input_node_ptr(0))) {
                // disable ConstantFolding also for Convert on zero_point
                // so we don't have to constantfold it and then convert it back to
                // low precision in LP transformations
                ov::disable_constant_folding(zero_point);
            }
        }

        // mark Multiply as dequantization node
        ov::mark_as_dequantization_node(multiply);

        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(root, "MarkDequantizationSubgraph");
    this->register_matcher(m, callback);
}
