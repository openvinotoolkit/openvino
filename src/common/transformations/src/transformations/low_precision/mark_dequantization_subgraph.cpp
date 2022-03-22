// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mark_dequantization_subgraph.hpp"

#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/rt_info/dequantization_node.hpp>
#include <transformations/rt_info/disable_constant_folding.hpp>

using namespace ngraph;

ngraph::pass::MarkDequantizationSubgraph::MarkDequantizationSubgraph(const element::TypeVector& precisions) {
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
    auto convert_pattern = ngraph::pattern::wrap_type<opset8::Convert>({input_pattern}, pattern::consumers_count(1));
    auto zero_point_pattern = pattern::any_input();
    auto subtract_pattern = ngraph::pattern::wrap_type<opset8::Subtract>({convert_pattern, zero_point_pattern});
    auto multiply_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({subtract_pattern, pattern::any_input()});
    auto multiply_no_subtract_pattern =
        ngraph::pattern::wrap_type<opset8::Multiply>({convert_pattern, pattern::any_input()});
    auto root = std::make_shared<pattern::op::Or>(OutputVector{multiply_pattern, multiply_no_subtract_pattern});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) -> bool {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& convert = pattern_map.at(convert_pattern).get_node_shared_ptr();
        const auto& input_precision = convert->get_input_element_type(0);

        // validation by Convert operation input precisions
        if (!precisions.empty()) {
            if (std::find(precisions.begin(), precisions.end(), input_precision) == precisions.end()) {
                return false;
            }
        }

        const auto& input = pattern_map.at(input_pattern);
        if (ov::is_type<opset8::Constant>(input.get_node())) {
            // disable ConstantFolding if dequantization subgraph is on constant data
            ov::disable_constant_folding(convert);
        }

        auto subtract_it = pattern_map.find(subtract_pattern);
        if (subtract_it != pattern_map.end()) {
            // mark Subtract as dequantization node
            ov::mark_as_dequantization_node(subtract_it->second.get_node_shared_ptr());
            auto zero_point = pattern_map.at(zero_point_pattern).get_node();
            if (ov::is_type<opset8::Convert>(zero_point) && input_precision == zero_point->get_input_element_type(0) &&
                ov::is_type<opset8::Constant>(zero_point->get_input_node_ptr(0))) {
                // disable ConstantFolding also for Convert on zero_point
                // so we don't have to constantfold it and then convert it back to
                // low precision in LP transformations
                ov::disable_constant_folding(zero_point->shared_from_this());
            }
        }

        // mark Multiply as dequantization node
        ov::mark_as_dequantization_node(m.get_match_root());

        return false;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(root, "MarkDequantizationSubgraph");
    this->register_matcher(m, callback);
}
