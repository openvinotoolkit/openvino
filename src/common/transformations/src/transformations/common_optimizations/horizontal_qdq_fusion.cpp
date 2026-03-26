// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/horizontal_qdq_fusion.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/subtract.hpp"
#include "transformations/pattern_blocks/qdq_block.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

HorizontalQDQFusion::HorizontalQDQFusion(const ov::element::TypeVector& supported_low_precisions,
                                         const ov::element::TypeVector& supported_original_precisions) {
    MATCHER_SCOPE(HorizontalQDQFusion);

    using namespace ov::pass::pattern;
    auto qdq_block =
        std::make_shared<op::QDQBlock>(type_matches_any(supported_original_precisions),
                                       type_matches_any(supported_low_precisions),
                                       type_matches_any(supported_original_precisions) && consumers_more_than(1));

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto dq_convert = qdq_block->get_anchor("dq_convert_pattern", pattern_map)->get_node_shared_ptr();
        const bool has_subtract = qdq_block->get_anchor("sub_pattern", pattern_map).has_value();
        const auto ref_zero_point =
            has_subtract ? qdq_block->get_anchor("sub_pattern", pattern_map)->get_node_shared_ptr()->input_value(1)
                         : ov::Output<ov::Node>{};

        auto ref_mul = qdq_block->get_anchor("mul_pattern", pattern_map)->get_node_shared_ptr();
        const auto ref_scale = *qdq_block->get_anchor("scale_pattern", pattern_map);

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
                if (!ov::op::util::outputs_are_equal(sub->input_value(1), ref_zero_point))
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

            if (!ov::op::util::outputs_are_equal(mul->input_value(1), ref_scale))
                continue;

            changed = changed || ov::replace_output_update_name(mul->output(0), ref_mul->output(0));
        }
        return changed;
    };

    auto m = std::make_shared<pattern::Matcher>(qdq_block, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace ov::pass
