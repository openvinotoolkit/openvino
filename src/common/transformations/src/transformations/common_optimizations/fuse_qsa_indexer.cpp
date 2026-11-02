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
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
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
using QSAIndexerInputIdx = op_internal::QSAIndexer::InputIdx;
using QSAIndexerOutputIdx = op_internal::QSAIndexer::OutputIdx;

FuseQSAIndexerBase::FuseQSAIndexerBase() {
    MATCHER_SCOPE(FuseQSAIndexerBase);
    using namespace ov::pass::pattern;

    constexpr size_t query_heads_axis = 1;
    constexpr size_t query_sequence_axis = 2;
    constexpr size_t query_head_dim_axis = 3;
    constexpr size_t keys_sequence_axis = 1;
    constexpr size_t rotary_dim_axis = 2;

    // QSAIndexer kernels are specialized for the query head count, the head size and the rotary dimension.
    // The ranks are fixed, since the callback reads the dimensions.
    // key_length is i32 as QSAIndexer expects: the plugins convert i64 to i32 before this pass.
    const auto p_query = any_input(rank_equals(4) && has_static_dims({query_heads_axis, query_head_dim_axis}));
    const auto p_raw_keys = any_input(rank_equals(3));
    const auto p_past_block_keys = any_input();
    const auto p_key_length = any_input(rank_equals(1) && type_matches(ov::element::i32));
    const auto p_rope_cos = any_input(rank_equals(3) && has_static_dim(rotary_dim_axis));
    const auto p_rope_sin = any_input(rank_equals(3));
    const auto p_k_norm_weight = any_input(rank_equals(1));
    const auto p_incomplete_raw_keys = any_input();
    const auto p_incomplete_rope = any_input();
    const auto p_prepare = wrap_type_strict_index<v17::QSAPrepare>({p_raw_keys,
                                                                    p_past_block_keys,
                                                                    p_key_length,
                                                                    p_rope_cos,
                                                                    p_rope_sin,
                                                                    p_k_norm_weight,
                                                                    p_incomplete_raw_keys,
                                                                    p_incomplete_rope});

    const auto p_indexer = wrap_type<v17::SparseAttentionIndexer>({p_query, p_prepare->output(0), p_key_length},
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
        // QSAIndexer takes the query and the raw keys for the same tokens, a single key_block_size and an enabled
        // key norm; the source ops validate the other head sizes against the query.
        if (!query.get_partial_shape()[query_sequence_axis].compatible(
                raw_keys.get_partial_shape()[keys_sequence_axis]) ||
            indexer->get_key_block_size() != prepare->get_key_block_size() ||
            !k_norm_weight.get_partial_shape()[0].compatible(head_dim)) {
            return false;
        }
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
        config.query_norm = false;
        config.query_rope = false;

        // QSAIndexer takes [B, L, H, D] activations and head-first states, while the public ops are head-first for
        // the query and head-less for the keys.
        ov::pass::NodeRegistry rg;
        const auto i64 = [&rg](const std::vector<int64_t>& values) {
            return rg.make<v0::Constant>(ov::element::i64, ov::Shape{values.size()}, values);
        };
        const ov::NodeVector sources{prepare, indexer};
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
        const auto key_length_i64 = rg.make<v0::Convert>(key_length, ov::element::i64);
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

        const auto query_blhd = rg.make<v1::Transpose>(query, i64({0, 2, 1, 3}));
        const auto fused = rg.make<op_internal::QSAIndexer>(ov::OutputVector{query_blhd,
                                                                             raw_keys_blhd,
                                                                             rope_for_batch(rope_cos),
                                                                             rope_for_batch(rope_sin),
                                                                             batched_key_length,
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

FuseQSAIndexerQueryRoPE::FuseQSAIndexerQueryRoPE() {
    MATCHER_SCOPE(FuseQSAIndexerQueryRoPE);
    using namespace ov::pass::pattern;

    const auto p_rope_input = any_input();
    const auto p_rope_cos = any_input();
    const auto p_rope_sin = any_input();
    const auto p_rope = wrap_type<op_internal::RoPE>(
        {p_rope_input,
         wrap_type<v0::Unsqueeze>({p_rope_cos, wrap_type<v0::Constant>(value_matches("1"))}),
         wrap_type<v0::Unsqueeze>({p_rope_sin, wrap_type<v0::Constant>(value_matches("1"))})},
        // QSAIndexer validates that its cos, shared with the RoPE, has indexer_rotary_dim elements.
        [](const ov::Output<ov::Node>& output) {
            const auto rope = ov::as_type<op_internal::RoPE>(output.get_node());
            auto rope_config = rope->get_config();
            rope_config.input_trans0213 = false;
            const auto& cos_shape = rope->get_input_partial_shape(1);
            return op_internal::is_plain_rotate_half(rope_config) && cos_shape.rank().is_static() &&
                   cos_shape[-1] == static_cast<int64_t>(rope_config.rotary_ndims);
        },
        {{"config.input_trans0213", true}, {"config.input_interleaved", false}, {"config.use_rope_cache", false}});
    const auto p_query_transpose =
        wrap_type<v1::Transpose>({p_rope, wrap_type<v0::Constant>(value_matches("0, 2, 1, 3"))});
    // FuseQSAIndexerBase broadcasts cos / sin to the batch unless it is proven equal.
    const auto p_indexer_rope_cos = optional<v3::Broadcast>({p_rope_cos, any_input()});
    const auto p_indexer_rope_sin = optional<v3::Broadcast>({p_rope_sin, any_input()});
    const auto p_indexer = wrap_type<op_internal::QSAIndexer>(
        {p_query_transpose,
         any_input(),
         p_indexer_rope_cos,
         p_indexer_rope_sin,
         any_input(),
         any_input(),
         any_input(),
         any_input(),
         any_input(),
         any_input()},
        [](const ov::Output<ov::Node>& output) {
            const auto& config = ov::as_type<op_internal::QSAIndexer>(output.get_node())->get_config();
            return !config.query_norm && !config.query_rope;
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto indexer = ov::as_type_ptr<op_internal::QSAIndexer>(pattern_map.at(p_indexer).get_node_shared_ptr());
        OPENVINO_ASSERT(indexer, "FuseQSAIndexerQueryRoPE: matched node is expected to be QSAIndexer");
        if (transformation_callback(indexer)) {
            return false;
        }
        auto config = indexer->get_config();
        config.query_rope = true;
        auto inputs = indexer->input_values();
        inputs[QSAIndexerInputIdx::INDEX_QUERY] = pattern_map.at(p_rope_input);
        const auto fused = std::make_shared<op_internal::QSAIndexer>(inputs, config);
        fused->set_friendly_name(indexer->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(p_rope).get_node_shared_ptr(), indexer}, fused);
        ov::replace_node(indexer, fused);
        register_new_node(fused);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_indexer, matcher_name), callback);
}

FuseQSAIndexerQueryNorm::FuseQSAIndexerQueryNorm() {
    MATCHER_SCOPE(FuseQSAIndexerQueryNorm);
    using namespace ov::pass::pattern;

    const auto p_raw_query = any_input();
    const auto p_rms = wrap_type<op_internal::RMS>({p_raw_query},
                                                   [](const ov::Output<ov::Node>& output) {
                                                       return output.get_element_type() ==
                                                              output.get_node()->get_input_element_type(0);
                                                   },
                                                   {{"axis", -1}, {"elementwise_affine", false}});
    const auto p_indexer = wrap_type<op_internal::QSAIndexer>(
        {p_rms,
         any_input(),
         any_input(),
         any_input(),
         any_input(),
         any_input(),
         any_input(),
         any_input(),
         any_input(),
         any_input()},
        [](const ov::Output<ov::Node>& output) {
            return !ov::as_type<op_internal::QSAIndexer>(output.get_node())->get_config().query_norm;
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto indexer = ov::as_type_ptr<op_internal::QSAIndexer>(pattern_map.at(p_indexer).get_node_shared_ptr());
        OPENVINO_ASSERT(indexer, "FuseQSAIndexerQueryNorm: matched node is expected to be QSAIndexer");
        if (transformation_callback(indexer)) {
            return false;
        }
        // The kernel normalizes the query and the keys with a single epsilon.
        const auto rms = ov::as_type_ptr<op_internal::RMS>(pattern_map.at(p_rms).get_node_shared_ptr());
        OPENVINO_ASSERT(rms, "FuseQSAIndexerQueryNorm: matched node is expected to be RMS");
        auto config = indexer->get_config();
        if (rms->get_epsilon() != config.epsilon) {
            return false;
        }
        config.query_norm = true;
        auto inputs = indexer->input_values();
        inputs[QSAIndexerInputIdx::INDEX_QUERY] = pattern_map.at(p_raw_query);
        const auto fused = std::make_shared<op_internal::QSAIndexer>(inputs, config);
        fused->set_friendly_name(indexer->get_friendly_name());
        ov::copy_runtime_info({rms, indexer}, fused);
        ov::replace_node(indexer, fused);
        register_new_node(fused);
        return true;
    };

    register_matcher(std::make_shared<Matcher>(p_indexer, matcher_name), callback);
}

}  // namespace ov::pass
