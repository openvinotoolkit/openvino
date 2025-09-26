// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "unsqueeze_broadcast_reshape_sdpa_fusion.hpp"

#include "intel_gpu/op/sdpa.hpp"
#include "intel_gpu/op/kv_cache.hpp"

#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/utils/utils.hpp"
#include "openvino/core/graph_util.hpp"

namespace ov::intel_gpu {
using ov::pass::pattern::op::Or;

UnsqueezeBroadcastReshapeSDPAFusion::UnsqueezeBroadcastReshapeSDPAFusion() {
    using namespace ov::pass::pattern;

    auto unsqueeze_predicate = rank_equals(5) && consumers_count(1);

    auto broadcast_predicate = unsqueeze_predicate && [](const ov::Output<ov::Node>& output) -> bool {
        const auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(output.get_node_shared_ptr());
        return broadcast && broadcast->get_broadcast_spec().m_type == ov::op::BroadcastType::BIDIRECTIONAL;
    };

    auto reshape_predicate = rank_equals(4) && consumers_count(1);

    auto input_a_m = any_input();
    auto input_attn_mask_m = any_input();
    auto input_scale_m = any_input();
    auto input_b_kvcache_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
    auto input_b_rope_m = wrap_type<ov::op::internal::RoPE>({any_input(), any_input(), any_input()});
    auto input_c_kvcache_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
    auto input_c_transpose_m = wrap_type<ov::op::v1::Transpose>({any_input(), any_input()});

    auto axes_const_b_m = wrap_type<ov::op::v0::Constant>();
    auto axes_const_c_m = wrap_type<ov::op::v0::Constant>();
    auto unsqueeze_b_m = wrap_type<ov::op::v0::Unsqueeze>({input_b_kvcache_m, axes_const_b_m}, unsqueeze_predicate);
    auto unsqueeze_c_m = wrap_type<ov::op::v0::Unsqueeze>({input_c_kvcache_m, axes_const_c_m}, unsqueeze_predicate);
    auto pre_reshape_b_m = wrap_type<ov::op::v1::Reshape>({input_b_rope_m, any_input()}, unsqueeze_predicate);
    auto pre_reshape_c_m = wrap_type<ov::op::v1::Reshape>({input_c_transpose_m, any_input()}, unsqueeze_predicate);

    auto broadcast_input_b_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{unsqueeze_b_m, pre_reshape_b_m});
    auto broadcast_input_c_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{unsqueeze_c_m, pre_reshape_c_m});
    auto broadcast_b_m = wrap_type<ov::op::v3::Broadcast>({broadcast_input_b_m, any_input()}, broadcast_predicate);
    auto broadcast_c_m = wrap_type<ov::op::v3::Broadcast>({broadcast_input_c_m, any_input()}, broadcast_predicate);
    auto reshape_b_m = wrap_type<ov::op::v1::Reshape>({broadcast_b_m, any_input()}, reshape_predicate);
    auto reshape_c_m = wrap_type<ov::op::v1::Reshape>({broadcast_c_m, any_input()}, reshape_predicate);

    auto convert_reshape_b_m = wrap_type<ov::op::v0::Convert>({reshape_b_m});
    auto reshape_b_input_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{reshape_b_m, convert_reshape_b_m});
    auto convert_reshape_c_m = wrap_type<ov::op::v0::Convert>({reshape_c_m});
    auto reshape_c_input_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{reshape_c_m, convert_reshape_c_m});

    auto sdpa_without_attn_mask_m = wrap_type<op::SDPA>({ input_a_m, reshape_b_m, reshape_c_m });
    auto sdpa_with_attn_mask_m = wrap_type<op::SDPA>({ input_a_m, reshape_b_input_m, reshape_c_input_m, input_attn_mask_m });
    auto sdpa_with_attn_mask_and_scale_m = wrap_type<op::SDPA>({ input_a_m, reshape_b_m, reshape_c_m, input_attn_mask_m, input_scale_m });

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();

        auto broadcast_b = ov::as_type_ptr<ov::op::v3::Broadcast>(pattern_map.at(broadcast_b_m).get_node_shared_ptr());
        auto broadcast_c = ov::as_type_ptr<ov::op::v3::Broadcast>(pattern_map.at(broadcast_c_m).get_node_shared_ptr());

        auto get_input_shape = [](const std::shared_ptr<ov::op::v3::Broadcast>& broadcast) -> std::vector<int32_t> {
            if (!broadcast) return {};
            auto input_node = broadcast->get_input_node_shared_ptr(0);
            if (input_node && input_node->get_output_partial_shape(0).is_static()) {
                auto pshape = input_node->get_output_shape(0);
                return std::vector<int32_t>(pshape.begin(), pshape.end());
            }
            return {};
        };

        auto valid_broadcast_target_shape = [](const std::vector<int32_t>& input_shape,
                                               const std::vector<int32_t>& target_shape,
                                               bool is_static_output) {
            if (is_static_output) {
                // For static output shapes, check that input_shape and target_shape differ in exactly one dimension
                if (input_shape.empty() || (input_shape.size() != target_shape.size())) return false;
                int diff_cnt = 0;
                for (size_t i = 0; i < input_shape.size(); ++i) {
                    if (input_shape[i] != target_shape[i]) ++diff_cnt;
                }
                return diff_cnt == 1;
            } else {
                // For dynamic output shapes, check the target_shape pattern
                return std::count_if(target_shape.begin(), target_shape.end(), [](int32_t s) { return s != 1; }) == 1;
            }
        };

        std::vector<int32_t> target_shape_val_b;
        auto target_shape_constant_b = ov::as_type_ptr<ov::op::v0::Constant>(broadcast_b->get_input_node_shared_ptr(1));
        if (target_shape_constant_b) {
            target_shape_val_b = target_shape_constant_b->cast_vector<int32_t>();
            std::vector<int32_t> input_shape_b = get_input_shape(broadcast_b);
            bool is_static_b = broadcast_b->get_output_partial_shape(0).is_static();
            if (!valid_broadcast_target_shape(input_shape_b, target_shape_val_b, is_static_b)) {
                return false;
            }
        }

        std::vector<int32_t> target_shape_val_c;
        auto target_shape_constant_c = ov::as_type_ptr<ov::op::v0::Constant>(broadcast_c->get_input_node_shared_ptr(1));
        if (target_shape_constant_c) {
            target_shape_val_c = target_shape_constant_c->cast_vector<int32_t>();
            std::vector<int32_t> input_shape_c = get_input_shape(broadcast_c);
            bool is_static_c = broadcast_c->get_output_partial_shape(0).is_static();
            if (!valid_broadcast_target_shape(input_shape_c, target_shape_val_c, is_static_c)) {
                return false;
            }
        }

        // Expect the same broadcast rules for key and value inputs
        if (target_shape_val_b != target_shape_val_c) {
            return false;
        }

        OutputVector data_inputs;
        data_inputs.push_back(pattern_map.at(input_a_m).get_node_shared_ptr());               // Q input
        if (pattern_map.find(input_b_kvcache_m) != pattern_map.end())
            data_inputs.push_back(pattern_map.at(input_b_kvcache_m).get_node_shared_ptr());   // K input from KVCache
        if (pattern_map.find(input_b_rope_m) != pattern_map.end())
            data_inputs.push_back(pattern_map.at(input_b_rope_m).get_node_shared_ptr());      // K input from RoPE
        if (pattern_map.find(input_c_kvcache_m) != pattern_map.end())
            data_inputs.push_back(pattern_map.at(input_c_kvcache_m).get_node_shared_ptr());   // V input from KVCache
        if (pattern_map.find(input_c_transpose_m) != pattern_map.end())
            data_inputs.push_back(pattern_map.at(input_c_transpose_m).get_node_shared_ptr()); // V input from Transpose

        auto sdpa = ov::as_type_ptr<op::SDPA>(m.get_match_root());
        if (pattern_map.find(sdpa_with_attn_mask_m) != pattern_map.end()) {
            data_inputs.push_back(sdpa->get_input_source_output(3)); // attn_mask
        } else if (pattern_map.find(sdpa_with_attn_mask_and_scale_m) != pattern_map.end()) {
            data_inputs.push_back(sdpa->get_input_source_output(3)); // attn_mask
            data_inputs.push_back(sdpa->get_input_source_output(4)); // scale
        }

        auto order_a = sdpa->get_input0_transpose_order();
        auto order_b = sdpa->get_input1_transpose_order();
        auto order_c = sdpa->get_input2_transpose_order();
        auto order_d = sdpa->get_output_transpose_order();

        auto sdpa_new = std::make_shared<op::SDPA>(data_inputs, sdpa->get_causal(), order_a, order_b, order_c, order_d);

        sdpa_new->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa_new);
        ov::replace_node(sdpa, sdpa_new);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "UnsqueezeBroadcastReshapeSDPAFusion");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
