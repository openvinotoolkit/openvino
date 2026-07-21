// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kvcache_gqa_broadcast_elimination.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/core/partial_shape.hpp"
#include "intel_gpu/op/sdpa.hpp"
#include <openvino/util/pp.hpp>

namespace ov::intel_gpu {
    KVCacheGQABroadcastElimination::KVCacheGQABroadcastElimination() {
        using namespace ov::pass::pattern;
        auto is_reshape_4d_to_5d = [](const ov::Node* node) {
            if (!ov::is_type<ov::op::v1::Reshape>(node)) {
                return false;
            }
            auto in_ps = node->get_input_partial_shape(0);
            auto out_ps = node->get_output_partial_shape(0);
            return in_ps.rank().is_static() && out_ps.rank().is_static() && in_ps.size() == 4 && out_ps.size() == 5;
        };

        auto reshape_5d_to_4d = [](const ov::Output<ov::Node>& output) {
            auto in_ps = output.get_node()->get_input_partial_shape(0);
            auto out_ps = output.get_node()->get_output_partial_shape(0);
            return in_ps.rank().is_static() && out_ps.rank().is_static() && in_ps.size() == 5 && out_ps.size() == 4;
        };

        auto expand_kv_group_4d_to_5d = [is_reshape_4d_to_5d](const ov::Output<ov::Node>& output) {
            if (auto concat = ov::as_type<ov::op::v0::Concat>(output.get_node())) {
                if (concat->get_input_size() < 2) {
                    return false;
                }
                auto first = concat->get_input_node_ptr(0);
                if (!is_reshape_4d_to_5d(first)) {
                    return false;
                }
                for (size_t i = 1; i < concat->get_input_size(); ++i) {
                    if (concat->get_input_node_ptr(i) != first) {
                        return false;
                    }
                }
                return true;
            }

            if (auto broadcast = ov::as_type<ov::op::v3::Broadcast>(output.get_node())) {
                return is_reshape_4d_to_5d(broadcast->get_input_node_ptr(0));
            }

            return false;
        };

        auto input_a_m = any_input();
        auto input_attn_mask_m = any_input();
        auto input_scale_m = any_input();

        auto expand_key_m = wrap_type<ov::op::v0::Concat, ov::op::v3::Broadcast>(expand_kv_group_4d_to_5d);

        auto reshape2_pattern_key_m = any_input();
        auto reshape_5d_to_4d_key_m = wrap_type<ov::op::v1::Reshape>({ expand_key_m, reshape2_pattern_key_m }, reshape_5d_to_4d);

        auto expand_value_m = wrap_type<ov::op::v0::Concat, ov::op::v3::Broadcast>(expand_kv_group_4d_to_5d);

        auto reshape2_pattern_value_m = any_input();
        auto reshape_5d_to_4d_value_m = wrap_type<ov::op::v1::Reshape>({ expand_value_m, reshape2_pattern_value_m }, reshape_5d_to_4d);

        auto sdpa_without_attn_mask_m = wrap_type<op::SDPA>({ input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m });
        auto sdpa_with_attn_mask_m =
            wrap_type<op::SDPA>({ input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m, any_input() });
        auto sdpa_with_attn_mask_and_scale_m =
            wrap_type<op::SDPA>({ input_a_m, reshape_5d_to_4d_key_m, reshape_5d_to_4d_value_m, input_attn_mask_m, input_scale_m });

        auto sdpa_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{ sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m });

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_value_map();
            auto sdpa = ov::as_type_ptr<op::SDPA>(m.get_match_root());
            if (pattern_map.count(expand_key_m) == 0 || pattern_map.count(expand_value_m) == 0) {
                return false;
            }
            if (pattern_map.count(expand_key_m) > 0) {
                auto key_source = pattern_map.at(expand_key_m).get_node()->input_value(0).get_node()->input_value(0);
                sdpa->input(1).replace_source_output(key_source);
            }
            if (pattern_map.count(expand_value_m) > 0) {
                auto value_source = pattern_map.at(expand_value_m).get_node()->input_value(0).get_node()->input_value(0);
                sdpa->input(2).replace_source_output(value_source);
            }
            return true;
        };
        auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "KVCacheGQABroadcastElimination");
        this->register_matcher(m, callback);
    }

}  // namespace ov::intel_gpu
