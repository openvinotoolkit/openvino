// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/horizontal_qdq_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

namespace {

// Check if two outputs represent the same constant value by constant-folding an Equal node.
bool outputs_are_equal(const ov::Output<ov::Node>& a, const ov::Output<ov::Node>& b) {
    auto eq = std::make_shared<ov::op::v1::Equal>(a, b);
    auto folded = ov::util::get_constant_from_source(eq);
    if (!folded)
        return false;
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(folded);
    if (!constant)
        return false;
    const auto& values = constant->get_vector<bool>();
    return std::all_of(values.begin(), values.end(), [](bool v) {
        return v;
    });
}

}  // namespace

namespace ov::pass {

HorizontalQDQFusion::HorizontalQDQFusion(const ov::element::TypeVector& supported_low_precisions,
                                         const ov::element::TypeVector& supported_original_precisions) {
    MATCHER_SCOPE(HorizontalQDQFusion);
    auto data_pattern = pattern::any_input(pattern::type_matches_any(supported_original_precisions));
    auto input_low_pattern = pattern::any_input();
    auto input_high_pattern = pattern::any_input();
    auto output_low_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto output_high_pattern = pattern::wrap_type<ov::op::v0::Constant>();
    auto fq_pattern = pattern::wrap_type<ov::op::v0::FakeQuantize>(
        {data_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});

    // convert1_pattern: Convert from original to low precision
    auto convert1_predicate = pattern::type_matches_any(supported_low_precisions);
    auto convert1_pattern = pattern::wrap_type<ov::op::v0::Convert>({fq_pattern}, convert1_predicate);

    auto dq_convert_predicate =
        pattern::type_matches_any(supported_original_precisions) && pattern::consumers_more_than(1);
    auto dq_convert_pattern = pattern::wrap_type<ov::op::v0::Convert>({convert1_pattern}, dq_convert_predicate);

    auto zero_point_pattern = pattern::any_input();
    auto sub_pattern = pattern::optional<ov::op::v1::Subtract>({dq_convert_pattern, zero_point_pattern});
    auto scale_pattern = pattern::any_input();
    auto mul_pattern = pattern::wrap_type<ov::op::v1::Multiply>({sub_pattern, scale_pattern});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto dq_convert = pattern_map.at(dq_convert_pattern).get_node_shared_ptr();
        const bool has_subtract = pattern_map.count(sub_pattern) != 0;
        const auto ref_zero_point =
            has_subtract ? pattern_map.at(sub_pattern).get_node_shared_ptr()->input_value(1) : ov::Output<ov::Node>{};

        // Extract reference constants from the pattern map.
        auto ref_mul = pattern_map.at(mul_pattern).get_node_shared_ptr();
        const auto ref_scale = pattern_map.at(scale_pattern);

        // Iterate over all consumers of dq_convert, check structure and constants against the
        // reference, and replace duplicate multiply outputs.
        bool changed = false;
        for (const auto& consumer_input : dq_convert->get_output_target_inputs(0)) {
            auto consumer = consumer_input.get_node()->shared_from_this();
            std::shared_ptr<ov::op::v1::Multiply> mul;

            if (has_subtract) {
                auto sub = ov::as_type_ptr<ov::op::v1::Subtract>(consumer);
                if (!sub || sub->get_input_node_shared_ptr(0) != dq_convert)
                    continue;
                if (!outputs_are_equal(sub->input_value(1), ref_zero_point))
                    continue;
                const auto sub_consumers = sub->get_output_target_inputs(0);
                if (sub_consumers.size() != 1)
                    continue;
                mul = ov::as_type_ptr<ov::op::v1::Multiply>(sub_consumers.begin()->get_node()->shared_from_this());
                if (!mul || mul->get_input_node_shared_ptr(0) != sub)
                    continue;
            } else {
                mul = ov::as_type_ptr<ov::op::v1::Multiply>(consumer);
                if (!mul || mul->get_input_node_shared_ptr(0) != dq_convert)
                    continue;
            }

            if (mul == ref_mul)
                continue;

            if (!outputs_are_equal(mul->input_value(1), ref_scale))
                continue;

            ov::replace_output_update_name(mul->output(0), ref_mul->output(0));
            changed = true;
        }

        return changed;
    };

    auto m = std::make_shared<pattern::Matcher>(mul_pattern, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
