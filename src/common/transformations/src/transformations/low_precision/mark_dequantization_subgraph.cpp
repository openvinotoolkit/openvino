// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mark_dequantization_subgraph.hpp"

#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::MarkDequantizationSubgraph::MarkDequantizationSubgraph(const element::TypeVector& precisions,
                                                                 const bool fold_subtract_const,
                                                                 const bool disable_fold_multiply_const) {
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
    auto convert_pattern = pattern::wrap_type<ov::op::v0::Convert>({input_pattern}, pattern::consumers_count(1));
    auto zero_point_pattern = pattern::any_input();
    auto subtract_pattern = pattern::wrap_type<ov::op::v1::Subtract>({convert_pattern, zero_point_pattern});
    auto multiply_pattern = pattern::wrap_type<ov::op::v1::Multiply>({subtract_pattern, pattern::any_input()});
    auto multiply_no_subtract_pattern =
        pattern::wrap_type<ov::op::v1::Multiply>({convert_pattern, pattern::any_input()});
    auto root = std::make_shared<pattern::op::Or>(OutputVector{multiply_pattern, multiply_no_subtract_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) -> bool {
        const auto& pattern_map = m.get_pattern_value_map();
        auto convert = pattern_map.at(convert_pattern).get_node_shared_ptr();
        auto input = pattern_map.at(input_pattern);
        const auto multiply = m.get_match_root();

        if (transformation_callback(multiply)) {
            return false;
        }

        auto subtract_it = pattern_map.find(subtract_pattern);
        if (subtract_it == pattern_map.end()) {
            for (size_t i = 0; i < multiply->get_input_size(); i++) {
                const auto node = ov::as_type_ptr<ov::op::v0::Convert>(multiply->get_input_node_shared_ptr(i));
                if (node && std::find(precisions.begin(), precisions.end(), node->get_input_element_type(0)) !=
                                precisions.end()) {
                    convert = node;
                    input = convert->input_value(0);
                }
            }
        }

        const auto& input_precision = input.get_element_type();
        // validation by Convert operation input precisions
        if (std::find(precisions.begin(), precisions.end(), input_precision) == precisions.end()) {
            return false;
        }

        if (ov::op::util::is_on_constant_path(input)) {
            // disable ConstantFolding if dequantization subgraph is on constant data
            ov::disable_constant_folding(convert);
            // It is also necessary to avoid precision conversion for constant nodes with input_precision
            auto keep_const_precision = [&](Node* node) {
                if (auto constant = ov::as_type<ov::op::v0::Constant>(node)) {
                    const auto& const_et = constant->get_element_type();
                    if (std::find(precisions.begin(), precisions.end(), const_et) != precisions.end())
                        ov::enable_keep_const_precision(convert->get_input_node_shared_ptr(0));
                }
            };
            std::unordered_set<Node*> visited;
            ov::op::util::visit_constant_path(input.get_node(), visited, keep_const_precision);
        }

        if (subtract_it != pattern_map.end()) {
            // mark Subtract as dequantization node
            ov::mark_as_dequantization_node(subtract_it->second.get_node_shared_ptr());
            auto zero_point = pattern_map.at(zero_point_pattern).get_node_shared_ptr();
            if (ov::is_type<ov::op::v0::Convert>(zero_point) &&
                input_precision == zero_point->get_input_element_type(0) &&
                ov::is_type<ov::op::v0::Constant>(zero_point->get_input_node_ptr(0))) {
                if (!fold_subtract_const) {
                    // disable ConstantFolding also for Convert on zero_point
                    // so we don't have to constantfold it and then convert it back to
                    // low precision in LP transformations
                    ov::disable_constant_folding(zero_point);
                    ov::enable_keep_const_precision(zero_point->get_input_node_shared_ptr(0));
                } else {
                    ov::enable_constant_folding(zero_point);
                    ov::disable_keep_const_precision(zero_point->get_input_node_shared_ptr(0));
                }
            }
        }

        // mark Multiply as dequantization node
        ov::mark_as_dequantization_node(multiply);
        auto scale = multiply->get_input_node_shared_ptr(1);
        if (ov::is_type<ov::op::v0::Convert>(scale) &&
            ov::is_type<ov::op::v0::Constant>(scale->get_input_node_ptr(0))) {
            if (disable_fold_multiply_const) {
                ov::disable_constant_folding(scale);
                ov::unmark_as_decompression(scale);
                ov::enable_keep_const_precision(scale->get_input_node_shared_ptr(0));
            }
        }

        return false;
    };

    auto m = std::make_shared<pattern::Matcher>(root, "MarkDequantizationSubgraph");
    this->register_matcher(m, callback);
}
