// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/paged_attention/paged_gated_delta_net_fusion.hpp"

#include <memory>
#include <string>
#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gated_delta_net.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/paged_gated_delta_net.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"

using ov::pass::pattern::any_input;
using ov::pass::pattern::wrap_type;

namespace v0 = ov::op::v0;

namespace {

constexpr const char* GATED_DELTA_STATE_TABLE_PREFIX = "gated_delta_state_table.";

std::string make_gated_delta_state_table_name(const size_t layer_index) {
    return std::string(GATED_DELTA_STATE_TABLE_PREFIX) + std::to_string(layer_index);
}

ov::PartialShape make_gated_delta_state_table_shape(const ov::PartialShape& state_shape) {
    // GDN input state shape is [B, H, D_k, D_v].
    // PagedGatedDeltaNet state table expects [?, H, D_v, D_k] — swap last two dims.
    if (state_shape.rank().is_static() && state_shape.rank().get_length() == 4) {
        return ov::PartialShape{ov::Dimension::dynamic(), state_shape[1], state_shape[3], state_shape[2]};
    }
    return ov::PartialShape::dynamic(4);
}

ov::Output<ov::Node> flatten_batch_length(const ov::Output<ov::Node>& input,
                                          const std::vector<int64_t>& tail_dim_indices) {
    // Flattens [B, L, ...tail_dims] to [B*L, ...tail_dims] by preserving tail_dim_indices.
    const auto shape_of = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    const auto idx_const = v0::Constant::create(ov::element::i64, ov::Shape{tail_dim_indices.size()}, tail_dim_indices);
    const auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    const auto tail_dims = std::make_shared<ov::op::v8::Gather>(shape_of, idx_const, axis_0);
    const auto flat_dim = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const auto flat_shape = std::make_shared<v0::Concat>(ov::OutputVector{flat_dim, tail_dims}, 0);
    const auto reshaped = std::make_shared<ov::op::v1::Reshape>(input, flat_shape, false);

    ov::copy_runtime_info(input.get_node_shared_ptr(), {shape_of, tail_dims, flat_shape, reshaped});

    return reshaped;
}
}  // namespace

namespace ov::pass {

PagedGatedDeltaNetFusion::PagedGatedDeltaNetFusion(ov::pass::paged_attention::PaParams& pa_params,
                                                   std::unordered_set<std::string>& var_ids_to_remove) {
    auto query = any_input();
    auto key = any_input();
    auto value = any_input();
    auto gate = any_input();
    auto beta = any_input();

    // recurrent_state must come from ReadValue(cache_param) directly or via Gather(ReadValue, beam_idx, axis).
    auto cache_param = any_input();
    auto read_value = wrap_type<ov::op::util::ReadValueBase>({cache_param});
    auto gathered_state = ov::pass::pattern::optional<ov::op::v8::Gather>({read_value, any_input(), any_input()});
    auto gdn = wrap_type<ov::op::internal::GatedDeltaNet>({query, key, value, gathered_state, gate, beta});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS, &pa_params, &var_ids_to_remove](
                                             ov::pass::pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        const auto& pm = m.get_pattern_value_map();
        const auto gdn_node = ov::as_type_ptr<ov::op::internal::GatedDeltaNet>(pm.at(gdn).get_node_shared_ptr());
        if (!gdn_node || gdn_node->get_output_size() != 2) {
            return false;
        }

        // Add la.* params lazily on first match — PaParams::add is idempotent.
        pa_params.add("subsequence_begins", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.block_indices", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.block_indices_begins", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.past_lens", ov::element::i32, ov::PartialShape{-1});
        pa_params.add("la.cache_interval", ov::element::i32, ov::PartialShape{-1});

        const auto state_consumers = gdn_node->output(1).get_target_inputs();
        const auto& state_out = pm.at(read_value);

        const auto state_table_param = pa_params.add(make_gated_delta_state_table_name(m_layer_index++),
                                                     state_out.get_element_type(),
                                                     make_gated_delta_state_table_shape(state_out.get_partial_shape()));
        enable_keep_const_precision(state_table_param);

        const auto rv = ov::as_type_ptr<ov::op::util::ReadValueBase>(pm.at(read_value).get_node_shared_ptr());
        OPENVINO_ASSERT(rv, "Matched cache node is expected to be ReadValue");
        var_ids_to_remove.insert(rv->get_variable_id());

        const auto query_flat = flatten_batch_length(pm.at(query), {2, 3});
        const auto key_flat = flatten_batch_length(pm.at(key), {2, 3});
        const auto value_flat = flatten_batch_length(pm.at(value), {2, 3});
        const auto gate_flat = flatten_batch_length(pm.at(gate), {2});
        const auto beta_flat = flatten_batch_length(pm.at(beta), {2});

        // Inputs 0-5 are flattened from matched GatedDeltaNet [B,L,H,*] to PagedGDN [B*L,H,*].
        const auto paged_gdn =
            std::make_shared<ov::op::internal::PagedGatedDeltaNet>(query_flat,
                                                                   key_flat,
                                                                   value_flat,
                                                                   state_table_param->output(0),
                                                                   gate_flat,
                                                                   beta_flat,
                                                                   pa_params["subsequence_begins"],
                                                                   pa_params["la.block_indices"],
                                                                   pa_params["la.block_indices_begins"],
                                                                   pa_params["la.past_lens"],
                                                                   pa_params["la.cache_interval"],
                                                                   gdn_node->get_fuse_qk_l2norm(),
                                                                   gdn_node->get_q_l2_norm_eps(),
                                                                   gdn_node->get_k_l2_norm_eps());

        paged_gdn->set_friendly_name(gdn_node->get_friendly_name() + "/PagedGatedDeltaNet");
        const auto query_shape = std::make_shared<ov::op::v3::ShapeOf>(pm.at(query), ov::element::i64);
        const auto value_shape = std::make_shared<ov::op::v3::ShapeOf>(pm.at(value), ov::element::i64);
        const auto axis_0 = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto idx_q = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 1, 2});
        const auto idx_v = v0::Constant::create(ov::element::i64, ov::Shape{1}, {3});
        const auto q_dims = std::make_shared<ov::op::v8::Gather>(query_shape, idx_q, axis_0);
        const auto v_dim = std::make_shared<ov::op::v8::Gather>(value_shape, idx_v, axis_0);
        const auto out0_shape = std::make_shared<v0::Concat>(ov::OutputVector{q_dims, v_dim}, 0);
        const auto paged_gdn_out = std::make_shared<ov::op::v1::Reshape>(paged_gdn, out0_shape, false);
        paged_gdn_out->set_friendly_name(gdn_node->get_friendly_name());

        ov::copy_runtime_info(gdn_node,
                              {paged_gdn, query_shape, value_shape, q_dims, v_dim, out0_shape, paged_gdn_out});

        // Disconnect GDN state output consumers; cleanup is driven by ReadValue variable ids.
        for (const auto& state_consumer : state_consumers) {
            // Reconnect consumer to the original state source so it becomes a dead branch.
            state_consumer.replace_source_output(gdn_node->input_value(3));
        }

        if (!ov::replace_output_update_name(gdn_node->output(0), paged_gdn_out->output(0))) {
            gdn_node->output(0).replace(paged_gdn_out->output(0));
        }

        register_new_node(paged_gdn_out);
        register_new_node(paged_gdn);
        return true;
    };

    const auto matcher = std::make_shared<ov::pass::pattern::Matcher>(gdn, "PagedGatedDeltaNetFusion");
    register_matcher(matcher, callback);
}

}  // namespace ov::pass
