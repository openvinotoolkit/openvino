// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/mark_dequantization_subgraph.hpp"

#include "itt.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass::pattern;

namespace {

ov::pass::pattern::op::Predicate check_precision(const ov::element::TypeVector& precisions) {
    return ov::pass::pattern::op::Predicate(
        [=](const Output<Node>& output) -> bool {
            return std::find(precisions.begin(), precisions.end(), output.get_element_type()) != precisions.end();
        },
        "check_precision");
}

using RTInfoSetter = std::function<void(const std::shared_ptr<ov::Node>& node)>;
void set_rt_info(const PatternValueMap& pt_map,
                 const RTInfoSetter& rt_info_setter,
                 const NodeVector& pattern_nodes,
                 const ov::element::TypeVector& precisions) {
    for (const auto& pattern_node : pattern_nodes) {
        if (pt_map.count(pattern_node)) {
            auto node = pt_map.at(pattern_node).get_node_shared_ptr();
            // we don't need to mark Converts with disable_cf attribute if the `from` type (input type)
            // is not in the `precisions` list.
            if (ov::as_type_ptr<v0::Convert>(node) && !check_precision(precisions)(node->input_value(0))) {
                continue;
            }

            rt_info_setter(node);
        }
    }
};

/*
In some cases we cannot swap Convert and Reshape because it would
lead to incorrect shapes: e.g.
If we have a following graph with Convert working for several Zero-Point Subgraphs,
we cannot perform swapping as this would break another part of the graph.

                               ZP Const
                                  │
                                  ▼
          Input                Convert                Input
            │                  │     │                  │
            ▼                  ▼     ▼                  ▼
 Scale      Convert      Reshape     Reshape      Convert      Scale
     |            │    (64,1,1,1)   (1,64,1,1)    │            │
     |            │      │                 │      │            |
     ▼            ▼      ▼                 ▼      ▼            ▼
     Reshape      Subtract                 Subtract      Reshape
           |      |                               |      |
           ▼      ▼                               ▼      ▼
           Multiply                               Multiply

Though, we can perform swapping if the shapes are same for all branches: e.g.

                               ZP Const
                                  │
                                  ▼
          Input                Convert                Input
            │                  │     │                  │
            ▼                  ▼     ▼                  ▼
            Convert      Reshape     Reshape      Convert
 Scale            │    (64,1,1,1)   (64,1,1,1)    │            Scale
     |            │      │                 │      │            |
     ▼            ▼      ▼                 ▼      ▼            ▼
     Reshape      Subtract                 Subtract      Reshape
           |      |                               |      |
           ▼      ▼                               ▼      ▼
           Multiply                               Multiply

Step 1: the left part of the graph would be matched transforming the graph above into the following form:

                      ZP Const
                         │
                         ▼
          Input       Reshape
            │        (64,1,1,1)
            │            │
            ▼            ▼
Scale       Convert   Convert          Input
    │          │      │     │            |
    ▼          ▼      ▼     ▼            ▼
    Reshape    Subtract     Reshape   Convert         Scale
          |      |         (64,1,1,1)    │            │
          ▼      ▼                │      │            |
          Multiply                ▼      ▼            ▼
                                  Subtract      Reshape
                                         |      |
                                         ▼      ▼
                                         Multiply

Step 2: the right part of the graph would be matched transforming the graph above into the final form:

                        ZP Const
                           │
                           ▼
                        Reshape
                       (64,1,1,1)
                           │
                           ▼
         Input          Reshape          Input
           │           (64,1,1,1)          │
           │               │               │
           ▼               ▼               ▼
Scale      Convert      Convert      Convert      Scale
    │            │      │     │      │            │
    ▼            ▼      ▼     ▼      ▼            ▼
    Reshape      Subtract     Subtract      Reshape
          |      |                   |      |
          ▼      ▼                   ▼      ▼
          Multiply                   Multiply

The double Reshapes are going to be folded in the next ConstantFolding

If there were 3 or even more branches with the same shapes, the process
will be the same: forthe first branch step 1 is applied, for all the
remainings step 2.
*/

bool can_swap(const PatternValueMap& pt_map,
              const std::shared_ptr<Node>& first_pattern,
              const std::shared_ptr<Node>& second_pattern) {
    if (pt_map.count(first_pattern) && pt_map.count(second_pattern)) {
        auto first_node = pt_map.at(first_pattern).get_node_shared_ptr();
        auto second_node = pt_map.at(second_pattern).get_node_shared_ptr();

        if (first_pattern->output(0).get_target_inputs().size() == 1)
            return true;

        auto target_inputs = first_node->output(0).get_target_inputs();

        if (target_inputs.begin()->get_node()->get_output_partial_shape(0).is_static()) {
            auto first_shape = target_inputs.begin()->get_node()->output(0).get_shape();

            // Step 1 (see steps description in the comments above)
            if (std::all_of(std::next(target_inputs.begin()),
                            target_inputs.end(),
                            [&](const ov::Input<ov::Node>& input) {
                                return input.get_node()->get_output_partial_shape(0).is_static() &&
                                       input.get_node()->get_shape() == first_shape;
                            })) {
                return true;
            } else if (first_node->get_output_partial_shape(0).is_static() &&
                       second_node->get_output_partial_shape(0).is_static() &&
                       first_node->get_output_shape(0) == second_node->get_output_shape(0)) {
                // Step 2
                return true;
            }
        }
    }

    return false;
}

bool swap_nodes(const PatternValueMap& pt_map,
                const std::shared_ptr<Node>& first,
                const std::shared_ptr<Node>& second) {
    if (can_swap(pt_map, first, second)) {
        auto first_node = pt_map.at(first).get_node_shared_ptr();
        auto second_node = pt_map.at(second).get_node_shared_ptr();
        auto target_inputs = second_node->output(0).get_target_inputs();
        second_node->input(0).replace_source_output(first_node->input_value(0));
        first_node->input(0).replace_source_output(second_node->output(0));
        for (const auto& in : target_inputs) {
            in.replace_source_output(first_node->output(0));
        }
        first_node->validate_and_infer_types();
        second_node->validate_and_infer_types();
        return true;
    }
    return false;
}

}  // namespace

ov::pass::MarkDequantization::MarkDequantization(const element::TypeVector& precisions,
                                                 const bool fold_subtract_const,
                                                 const bool fold_multiply_const) {
    MATCHER_SCOPE(MarkDequantization);

    // data input:
    auto input_pattern = any_input(check_precision(precisions));
    auto convert_pattern = wrap_type<v0::Convert>({input_pattern}, consumers_count(1));

    // zero points:
    auto zp_pattern = any_input();
    auto zp_convert_pattern = pattern::optional<v0::Convert>(zp_pattern);
    auto zp_reshape_pattern = pattern::optional<v1::Reshape, v0::Unsqueeze>({zp_convert_pattern, any_input()});
    auto subtract_pattern = pattern::optional<v1::Subtract>({convert_pattern, zp_reshape_pattern});

    // scale:
    auto scale_pattern = any_input();
    auto scale_convert_pattern = pattern::optional<v0::Convert>(scale_pattern);
    auto scale_reshape_pattern = pattern::optional<v1::Reshape, v0::Unsqueeze>({scale_convert_pattern, any_input()});
    auto multiply_pattern = wrap_type<v1::Multiply>({subtract_pattern, scale_reshape_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) -> bool {
        const auto& pt_map = m.get_pattern_value_map();
        auto convert = pt_map.at(convert_pattern);
        auto input = pt_map.at(input_pattern);
        const auto multiply = m.get_match_root();

        if (transformation_callback(multiply)) {
            return false;
        }

        // Multiply and Subtract have to be marked as dq
        set_rt_info(pt_map, mark_as_dequantization_node, {subtract_pattern, multiply_pattern}, {/* not applicable */});

        // Convert might be presented on scales, zp and data_input.
        // Depending on the transformation arguments they have to be marked/unmarked with disable_cf rt_info.
        NodeVector converts_to_mark = {convert_pattern};
        NodeVector converts_to_unmark = {};

        if (fold_subtract_const) {
            converts_to_unmark.push_back(zp_convert_pattern);
        } else {
            converts_to_mark.push_back(zp_convert_pattern);
        }

        if (fold_multiply_const) {
            converts_to_unmark.push_back(scale_convert_pattern);
        } else {
            converts_to_mark.push_back(scale_convert_pattern);
        }

        set_rt_info(pt_map, disable_constant_folding, converts_to_mark, precisions);
        set_rt_info(pt_map, enable_constant_folding, converts_to_unmark, precisions);

        // Move Reshape/Unsqueeze ops up to fold them in ConstantFolding.
        auto changed = swap_nodes(pt_map, zp_convert_pattern, zp_reshape_pattern);
        changed = swap_nodes(pt_map, scale_convert_pattern, scale_reshape_pattern) || changed;
        return changed;
    };

    auto m = std::make_shared<Matcher>(multiply_pattern, "MarkDequantization");
    this->register_matcher(m, callback);
}

ov::pass::KeepConstPrecision::KeepConstPrecision(const element::TypeVector& precisions,
                                                 bool fold_subtract_const,
                                                 bool fold_multiply_const) {
    MATCHER_SCOPE(KeepConstPrecision);

    // data input:
    auto input_pattern = any_input();
    auto convert_pattern = wrap_type<v0::Convert>({input_pattern}, consumers_count(1));

    // zero points:
    auto zp_pattern = any_input();
    auto zp_convert_pattern = pattern::optional<v0::Convert>(zp_pattern);
    auto zp_reshape_pattern = pattern::optional<v1::Reshape, v0::Unsqueeze>({zp_convert_pattern, any_input()});
    auto subtract_pattern = pattern::optional<v1::Subtract>({convert_pattern, zp_reshape_pattern});

    // scale:
    auto scale_pattern = any_input();
    auto scale_convert_pattern = pattern::optional<v0::Convert>(scale_pattern);
    auto scale_reshape_pattern = pattern::optional<v1::Reshape, v0::Unsqueeze>({scale_convert_pattern, any_input()});
    auto multiply_pattern = wrap_type<v1::Multiply>({subtract_pattern, scale_reshape_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) -> bool {
        const auto& pt_map = m.get_pattern_value_map();
        const auto multiply = m.get_match_root();

        if (transformation_callback(multiply)) {
            return false;
        }

        using PatternNode = std::shared_ptr<Node>;
        std::map<PatternNode, bool> keep_const_precisions = {{input_pattern, false},
                                                             {zp_pattern, fold_subtract_const},
                                                             {scale_pattern, fold_multiply_const}};
        for (const auto& pattern_node : keep_const_precisions) {
            if (pt_map.count(pattern_node.first)) {
                auto node = pt_map.at(pattern_node.first).get_node_shared_ptr();
                if (ov::as_type_ptr<v0::Constant>(node) && check_precision(precisions)(node->output(0))) {
                    if (pattern_node.second) {
                        ov::disable_keep_const_precision(node);
                    } else {
                        ov::enable_keep_const_precision(node);
                    }
                }
            }
        }
        return false;
    };

    auto m = std::make_shared<Matcher>(multiply_pattern, "KeepConstPrecision");
    this->register_matcher(m, callback);
}
