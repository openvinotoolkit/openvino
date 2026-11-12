// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"

#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::pass {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v17 = ov::op::v17;
namespace op_internal = ov::op::internal;
using QSAIndexerOutputIdx = op_internal::QSAIndexer::OutputIdx;

FuseQSAIndexer::FuseQSAIndexer() {
    MATCHER_SCOPE(FuseQSAIndexer);
    using namespace ov::pass::pattern;

    constexpr size_t query_heads_axis = 2;
    constexpr size_t query_sequence_axis = 1;
    constexpr size_t query_head_dim_axis = 3;
    constexpr size_t keys_sequence_axis = 1;
    constexpr size_t rotary_dim_axis = 2;

    // QSAIndexer kernels are specialized for the query head count, the head size and the rotary dimension.
    // The ranks are fixed, since the callback reads the dimensions.
    const auto p_query = any_input(rank_equals(4) && has_static_dims({query_heads_axis, query_head_dim_axis}));
    const auto p_raw_keys = any_input(rank_equals(3));
    const auto p_past_block_keys = any_input();
    const auto p_key_length = any_input(rank_equals(1));
    const auto p_rope_cos = any_input(rank_equals(3) && has_static_dim(rotary_dim_axis));
    const auto p_rope_sin = any_input(rank_equals(3));
    const auto p_k_norm_weight = any_input(rank_equals(1));
    const auto p_incomplete_raw_keys = any_input();
    const auto p_incomplete_rope = any_input();
    const auto p_rms = wrap_type<op_internal::RMS>({p_query},
                                                   [](const ov::Output<ov::Node>& output) {
                                                       return output.get_element_type() ==
                                                              output.get_node()->get_input_element_type(0);
                                                   },
                                                   {{"axis", -1}, {"elementwise_affine", false}});
    // NopElimination can replace the head-axis Unsqueeze of [tokens, 1, D] with a Reshape.
    const auto with_head_axis = [](const ov::Output<ov::Node>& rows) {
        const auto unsqueeze = wrap_type<v0::Unsqueeze>({rows, wrap_type<v0::Constant>(value_matches("1"))});
        const auto reshape = wrap_type<v1::Reshape>({rows, wrap_const()}, [](const ov::Output<ov::Node>& output) {
            const auto& shape = output.get_node()->get_input_partial_shape(0);
            return shape.rank() == 3 && shape[1] == 1 && shape[2].is_static() && shape[2].get_length() > 0 &&
                   output.get_partial_shape() == ov::PartialShape{shape[0], 1, 1, shape[2]};
        });
        return unsqueeze | reshape;
    };
    const auto p_rope = wrap_type<op_internal::RoPE>(
        {p_rms, with_head_axis(p_rope_cos), with_head_axis(p_rope_sin)},
        [](const ov::Output<ov::Node>& output) {
            const auto rope = ov::as_type<op_internal::RoPE>(output.get_node());
            auto config = rope->get_config();
            config.input_trans0213 = false;
            const auto& cos_shape = rope->get_input_partial_shape(1);
            return op_internal::is_plain_rotate_half(config) && cos_shape.rank() == 4 &&
                   cos_shape[-1] == static_cast<int64_t>(config.rotary_ndims);
        },
        {{"config.input_trans0213", true}, {"config.input_interleaved", false}, {"config.use_rope_cache", false}});
    const auto p_prepare = wrap_type_strict_index<v17::QSAPrepare>({p_raw_keys,
                                                                    p_past_block_keys,
                                                                    p_key_length,
                                                                    p_rope_cos,
                                                                    p_rope_sin,
                                                                    p_k_norm_weight,
                                                                    p_incomplete_raw_keys,
                                                                    p_incomplete_rope});

    const auto p_indexer = wrap_type<v17::SparseAttentionIndexer>(
        {p_rope, p_prepare->output(0), p_key_length},
        [=](const ov::Output<ov::Node>& output) {
            const auto indexer = ov::as_type<v17::SparseAttentionIndexer>(output.get_node());
            const auto prepare = ov::as_type<v17::QSAPrepare>(indexer->get_input_node_ptr(1));
            const auto rope = ov::as_type<op_internal::RoPE>(indexer->get_input_node_ptr(0));
            if (!prepare || !rope) {
                return false;
            }
            const auto rms = ov::as_type<op_internal::RMS>(rope->get_input_node_ptr(0));
            if (!rms) {
                return false;
            }
            const auto& query_shape = rms->get_input_partial_shape(0);
            const auto& keys_shape = prepare->get_input_partial_shape(0);
            const auto& weight_shape = prepare->get_input_partial_shape(5);
            return query_shape.rank() == 4 && keys_shape.rank() == 3 && weight_shape.rank() == 1 &&
                   query_shape[query_sequence_axis].compatible(keys_shape[keys_sequence_axis]) &&
                   weight_shape[0].compatible(query_shape[query_head_dim_axis]) &&
                   indexer->get_key_block_size() == prepare->get_key_block_size() &&
                   rms->get_epsilon() == prepare->get_epsilon();
        },
        {{"num_selection_heads", 1}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto indexer =
            ov::as_type_ptr<v17::SparseAttentionIndexer>(pattern_map.at(p_indexer).get_node_shared_ptr());
        OPENVINO_ASSERT(indexer, "FuseQSAIndexer: matched node is expected to be SparseAttentionIndexer");
        if (transformation_callback(indexer)) {
            return false;
        }
        const auto prepare = ov::as_type_ptr<v17::QSAPrepare>(m.get_pattern_map().at(p_prepare));
        OPENVINO_ASSERT(prepare, "FuseQSAIndexer: matched node is expected to be QSAPrepare");

        const auto& query = pattern_map.at(p_query);
        const auto& raw_keys = pattern_map.at(p_raw_keys);
        const auto& k_norm_weight = pattern_map.at(p_k_norm_weight);
        const auto& head_dim = query.get_partial_shape()[query_head_dim_axis];
        const auto& past_block_keys = pattern_map.at(p_past_block_keys);
        const auto& key_length = pattern_map.at(p_key_length);
        const auto& incomplete_raw_keys = pattern_map.at(p_incomplete_raw_keys);
        const auto& incomplete_rope = pattern_map.at(p_incomplete_rope);
        const auto& rope_cos = pattern_map.at(p_rope_cos);
        const auto& rope_sin = pattern_map.at(p_rope_sin);

        // The source ops validate the batch, the head size, element types and token_budget % key_block_size == 0.
        // scale stays 0, the default 1 / sqrt(indexer_head_dim).
        const auto key_block_size = prepare->get_key_block_size();
        const auto data_type = raw_keys.get_element_type();
        const auto& batch_dim = raw_keys.get_partial_shape()[0];
        op_internal::QSAIndexer::Config config;
        config.key_block_size = static_cast<size_t>(key_block_size);
        config.block_topk = static_cast<size_t>(indexer->get_token_budget() / key_block_size);
        config.indexer_rotary_dim = static_cast<size_t>(rope_cos.get_partial_shape()[rotary_dim_axis].get_length());
        config.indexer_head_dim = static_cast<size_t>(head_dim.get_length());
        config.epsilon = prepare->get_epsilon();

        // QSAIndexer takes [B, L, H, D] activations and head-first states, while the public ops are head-first for
        // the query and head-less for the keys.
        ov::pass::NodeRegistry rg;
        const auto i64 = [&rg](const std::vector<int64_t>& values) {
            return rg.make<v0::Constant>(ov::element::i64, ov::Shape{values.size()}, values);
        };
        const ov::NodeVector sources{prepare,
                                     indexer,
                                     pattern_map.at(p_rms).get_node_shared_ptr(),
                                     pattern_map.at(p_rope).get_node_shared_ptr()};
        const auto dims_of = [&sources](const ov::Output<ov::Node>& data, const std::vector<size_t>& axes) {
            return ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, axes, sources);
        };

        const auto q_norm_weight =
            rg.make<v0::Constant>(data_type, ov::Shape{config.indexer_head_dim}, std::vector<float>{1.0f});
        const auto activation_head_axis = i64({2});
        const auto state_head_axis = i64({1});
        const auto raw_keys_blhd = rg.make<v0::Unsqueeze>(raw_keys, activation_head_axis);
        const auto incomplete_raw_keys_with_head = rg.make<v0::Unsqueeze>(incomplete_raw_keys, state_head_axis);
        // The Broadcasts below are skipped only when the batches are proven equal by static values.
        const auto is_batch = [&batch_dim](const ov::Dimension& dim) {
            return dim.is_static() && batch_dim.is_static() && dim.get_length() == batch_dim.get_length();
        };
        // key_length is [N] or [1].
        ov::Output<ov::Node> batched_key_length = key_length;
        if (!is_batch(key_length.get_partial_shape()[0])) {
            batched_key_length = rg.make<v3::Broadcast>(key_length, dims_of(raw_keys, {0}));
        }

        // QSAPrepare broadcasts cos / sin of a single row over the batch, while QSAIndexer takes them per row.
        const auto rope_for_batch = [&](const ov::Output<ov::Node>& rope) -> ov::Output<ov::Node> {
            if (is_batch(rope.get_partial_shape()[0])) {
                return rope;
            }
            const auto target = rg.add(ov::op::util::make_try_fold<v0::Concat>(
                ov::OutputVector{dims_of(raw_keys, {0, keys_sequence_axis}),
                                 i64({static_cast<int64_t>(config.indexer_rotary_dim)})},
                0));
            return rg.make<v3::Broadcast>(rope, target);
        };

        // QSAPrepare output 0 holds the complete blocks of the longest sequence, while QSAIndexer keeps the cache
        // shape, so the cache is grown by zero blocks before the call.
        const auto key_length_i64 = key_length.get_element_type() == ov::element::i64
                                        ? key_length
                                        : rg.make<v0::Convert>(key_length, ov::element::i64);
        const auto max_key_length = rg.make<v1::ReduceMax>(key_length_i64, i64({0}), true);
        const auto needed_blocks = rg.make<v1::Divide>(max_key_length, i64({key_block_size}));
        const auto past_block_count = dims_of(past_block_keys, {1});
        const auto missing_blocks = rg.make<v1::Subtract>(needed_blocks, past_block_count);
        const auto cache_growth_blocks = rg.make<v1::Maximum>(missing_blocks, i64({0}));
        const auto cache_growth_shape = rg.add(ov::op::util::make_try_fold<v0::Concat>(
            ov::OutputVector{dims_of(raw_keys, {0}),
                             i64({1}),
                             cache_growth_blocks,
                             i64({static_cast<int64_t>(config.indexer_head_dim)})},
            0));
        const auto zero = rg.make<v0::Constant>(data_type, ov::Shape{}, std::vector<float>{0.0f});
        const auto cache_padding = rg.make<v3::Broadcast>(zero, cache_growth_shape);
        const auto past_block_keys_with_head = rg.make<v0::Unsqueeze>(past_block_keys, state_head_axis);
        constexpr int64_t state_blocks_axis = 2;
        const auto block_keys_cache =
            rg.make<v0::Concat>(ov::OutputVector{past_block_keys_with_head, cache_padding}, state_blocks_axis);

        const auto key_length_i32 = key_length.get_element_type() == ov::element::i32
                                        ? batched_key_length
                                        : rg.make<v0::Convert>(batched_key_length, ov::element::i32);
        const auto fused = rg.make<op_internal::QSAIndexer>(ov::OutputVector{query,
                                                                             raw_keys_blhd,
                                                                             rope_for_batch(rope_cos),
                                                                             rope_for_batch(rope_sin),
                                                                             key_length_i32,
                                                                             q_norm_weight,
                                                                             k_norm_weight,
                                                                             block_keys_cache,
                                                                             incomplete_raw_keys_with_head,
                                                                             incomplete_rope},
                                                            config);
        fused->set_friendly_name(indexer->get_friendly_name());
        register_new_node(fused);
        // The squeezed states replace QSAPrepare outputs 0 and 1, so they are named after them.
        const auto squeeze_state = [&](size_t fused_output, size_t prepare_output) {
            const auto squeeze = rg.make<v0::Squeeze>(fused->output(fused_output), state_head_axis);
            squeeze->set_friendly_name(prepare->get_friendly_name() + "." + std::to_string(prepare_output));
            return squeeze;
        };
        const auto block_keys = squeeze_state(QSAIndexerOutputIdx::BLOCK_INDEX_KEYS_CACHE, 0);
        const auto incomplete_raw_keys_out = squeeze_state(QSAIndexerOutputIdx::PAST_INCOMPLETE_RAW_K, 1);
        ov::copy_runtime_info(sources, rg.get());

        // replace_node/replace_output_update_name do not support replacing two multi-output producers with one node.
        // Output::replace moves consumers, tensor names, and output runtime info.
        indexer->output(0).replace(fused->output(QSAIndexerOutputIdx::SELECTED_BLOCK_INDICES));
        indexer->output(1).replace(fused->output(QSAIndexerOutputIdx::SELECTED_BLOCK_COUNT));
        prepare->output(0).replace(block_keys);
        prepare->output(1).replace(incomplete_raw_keys_out);
        prepare->output(2).replace(fused->output(QSAIndexerOutputIdx::PAST_INCOMPLETE_ROPE));
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_indexer, matcher_name), callback);
}

}  // namespace ov::pass
