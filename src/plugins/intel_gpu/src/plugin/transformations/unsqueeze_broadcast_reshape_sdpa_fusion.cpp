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
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {
using ov::pass::pattern::op::Or;

UnsqueezeBroadcastReshapeSDPAFusion::UnsqueezeBroadcastReshapeSDPAFusion() {
    using namespace ov::pass::pattern;

    auto not_reshape = [](const ov::Output<ov::Node>& output) -> bool {
        return std::dynamic_pointer_cast<ov::op::v1::Reshape>(output.get_node_shared_ptr()) == nullptr;
    };

    auto unsqueeze_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        return rank_equals(5)(output) && consumers_count(1);
    };

    auto broadcast_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        const auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(output.get_node_shared_ptr());
        if (!broadcast || broadcast->get_broadcast_spec().m_type != ov::op::BroadcastType::BIDIRECTIONAL)
            return false;
        return rank_equals(5)(output) && consumers_count(1);
    };

    auto reshape_predicate = [](const ov::Output<ov::Node>& output) -> bool {
        return rank_equals(4)(output) && consumers_count(1);
    };

    auto input_a_m = any_input(not_reshape);
    auto input_attn_mask = any_input();
    auto input_scale = any_input();
    auto input_b_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
    auto input_c_m = wrap_type<ov::intel_gpu::op::KVCache>({any_input(), any_input()});
    auto axes_const_b_m = wrap_type<ov::op::v0::Constant>();
    auto axes_const_c_m = wrap_type<ov::op::v0::Constant>();
    auto unsqueeze_b_m = wrap_type<ov::op::v0::Unsqueeze>({input_b_m, axes_const_b_m}, unsqueeze_predicate);
    auto unsqueeze_c_m = wrap_type<ov::op::v0::Unsqueeze>({input_c_m, axes_const_c_m}, unsqueeze_predicate);
    auto broadcast_b_m = wrap_type<ov::op::v3::Broadcast>({unsqueeze_b_m, any_input()}, broadcast_predicate);
    auto broadcast_c_m = wrap_type<ov::op::v3::Broadcast>({unsqueeze_c_m, any_input()}, broadcast_predicate);
    auto reshape_b_m = wrap_type<ov::op::v1::Reshape>({broadcast_b_m, any_input()}, reshape_predicate);
    auto reshape_c_m = wrap_type<ov::op::v1::Reshape>({broadcast_c_m, any_input()}, reshape_predicate);

    auto sdpa_without_attn_mask_m = wrap_type<op::SDPA>({ input_a_m, reshape_b_m, reshape_c_m });
    auto sdpa_with_attn_mask_m = wrap_type<op::SDPA>({ input_a_m, reshape_b_m, reshape_c_m, input_attn_mask });
    auto sdpa_with_attn_mask_and_scale_m = wrap_type<op::SDPA>({ input_a_m, reshape_b_m, reshape_c_m, input_attn_mask, input_scale });

    auto sdpa_m = std::make_shared<Or>(OutputVector{sdpa_without_attn_mask_m, sdpa_with_attn_mask_m, sdpa_with_attn_mask_and_scale_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        const auto& pattern_map = m.get_pattern_value_map();

        auto valid_broadcast_target_shape = [](const std::vector<int32_t>& target_shape) {
            return std::count_if(target_shape.begin(), target_shape.end(), [](int32_t s) { return s != 1; }) == 1;
        };
        auto broadcast_b = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(pattern_map.at(broadcast_b_m).get_node_shared_ptr());
        auto broadcast_c = std::dynamic_pointer_cast<ov::op::v3::Broadcast>(pattern_map.at(broadcast_c_m).get_node_shared_ptr());

        std::vector<int32_t> target_shape_val_b;
        auto target_shape_constant_b = std::dynamic_pointer_cast<ov::op::v0::Constant>(broadcast_c->get_input_node_shared_ptr(1));
        if (target_shape_constant_b) {
            target_shape_val_b = target_shape_constant_b->cast_vector<int32_t>();
            if (!valid_broadcast_target_shape(target_shape_val_b)) {
                return false;
            }
        }

        std::vector<int32_t> target_shape_val_c;
        auto target_shape_constant_c = std::dynamic_pointer_cast<ov::op::v0::Constant>(broadcast_b->get_input_node_shared_ptr(1));
        if (target_shape_constant_c) {
            target_shape_val_c = target_shape_constant_c->cast_vector<int32_t>();
            if (!valid_broadcast_target_shape(target_shape_val_c)) {
                return false;
            }
        }

        // Expect the same broadcast rules for key and value inputs
        if (target_shape_val_b != target_shape_val_c) {
            return false;
        }

        OutputVector data_inputs;
        data_inputs.push_back(pattern_map.at(input_a_m).get_node_shared_ptr()); // Q
        data_inputs.push_back(pattern_map.at(input_b_m).get_node_shared_ptr()); // K
        data_inputs.push_back(pattern_map.at(input_c_m).get_node_shared_ptr()); // V

        auto sdpa = std::dynamic_pointer_cast<op::SDPA>(m.get_match_root());
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

        auto sdpa_new = std::make_shared<op::SDPA>(data_inputs, sdpa->get_causal(), false /* is_kv_compressed*/, order_a, order_b, order_c, order_d);

        sdpa_new->set_friendly_name(sdpa->get_friendly_name());
        ov::copy_runtime_info(m.get_matched_nodes(), sdpa_new);
        ov::replace_node(sdpa, sdpa_new);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdpa_m, "UnsqueezeBroadcastReshapeSDPAFusion");
    this->register_matcher(m, callback);
}

}  // namespace intel_gpu
}  // namespace ov
