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
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/rt_info/disable_constant_folding.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::pass::pattern;

namespace {

bool check_precision(const ov::element::Type_t type_to_check, const ov::element::TypeVector& precisions) {
    return std::find(precisions.begin(), precisions.end(), type_to_check) != precisions.end();
};

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
            if (ov::as_type_ptr<v0::Convert>(node) && !check_precision(node->get_input_element_type(0), precisions)) {
                continue;
            }

            rt_info_setter(node);
        }
    }
};

bool swap_nodes(const PatternValueMap& pt_map,
                const std::shared_ptr<Node>& first,
                const std::shared_ptr<Node>& second) {
    if (pt_map.count(first) && pt_map.count(second)) {
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
        auto convert = pt_map.at(convert_pattern);
        auto input = pt_map.at(input_pattern);
        const auto multiply = m.get_match_root();

        if (!check_precision(input.get_element_type(), precisions) || transformation_callback(multiply)) {
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

ov::pass::KeepConstsPrecision::KeepConstsPrecision(const element::TypeVector& precisions,
                                                   bool fold_subtract_const,
                                                   bool fold_multiply_const) {
    MATCHER_SCOPE(KeepConstsPrecision);

    // data input:
    auto input_pattern = any_input();
    auto convert_pattern = wrap_type<v0::Convert>({input_pattern}, consumers_count(1));

    // zero points:
    auto zp_pattern = any_input();
    auto zp_convert_pattern = pattern::optional<v0::Convert>(zp_pattern);
    auto subtract_pattern = pattern::optional<v1::Subtract>({convert_pattern, zp_convert_pattern});

    // scale:
    auto scale_pattern = any_input();
    auto scale_convert_pattern = pattern::optional<v0::Convert>(scale_pattern);
    auto multiply_pattern = wrap_type<v1::Multiply>({subtract_pattern, scale_convert_pattern});

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
                const auto& precision = node->get_output_element_type(0);
                if (ov::as_type_ptr<v0::Constant>(node) && check_precision(precision, precisions)) {
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

    auto m = std::make_shared<Matcher>(multiply_pattern, "KeepConstsPrecision");
    this->register_matcher(m, callback);
}
