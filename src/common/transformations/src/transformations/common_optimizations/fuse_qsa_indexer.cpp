// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/fuse_qsa_indexer.hpp"

#include <algorithm>
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
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/qsa_indexer.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/node_registry.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
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
namespace v8 = ov::op::v8;
namespace op_internal = ov::op::internal;
using QSAIndexerOutputIdx = op_internal::QSAIndexer::OutputIdx;

FuseQSAIndexer::FuseQSAIndexer(bool padding) {
    MATCHER_SCOPE(FuseQSAIndexer);
    using namespace ov::pass::pattern;

    constexpr size_t query_heads_axis = 1;
    constexpr size_t query_sequence_axis = 2;
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
        const auto key_block_size = prepare->get_key_block_size();
        const auto data_type = raw_keys.get_element_type();
        const auto& batch_dim = raw_keys.get_partial_shape()[0];
        op_internal::QSAIndexer::Config config;
        config.key_block_size = static_cast<size_t>(key_block_size);
        config.block_topk = static_cast<size_t>(indexer->get_token_budget() / key_block_size);
        config.indexer_rotary_dim = static_cast<size_t>(rope_cos.get_partial_shape()[rotary_dim_axis].get_length());
        config.indexer_head_dim = static_cast<size_t>(head_dim.get_length());
        config.epsilon = prepare->get_epsilon();
        config.scale = op_internal::QSAIndexer::effective_scale(config);
        config.query_norm = false;
        config.query_rope = false;

        // QSAIndexer takes [B, L, H, D] activations and head-first states, while the public ops are head-first for
        // the query and head-less for the keys.
        ov::pass::NodeRegistry rg;
        const ov::NodeVector sources{prepare, indexer};
        // The shape and length subgraphs are built in i32: plugins may run this pass after lowering i64, so i64
        // tensors created here could reach kernels or shape subgraphs that support only i32.
        const auto dims_of = [&sources](const ov::Output<ov::Node>& data, const std::vector<size_t>& axes) {
            return ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(data,
                                                                                      axes,
                                                                                      sources,
                                                                                      ov::element::i32);
        };

        const auto q_norm_weight =
            rg.make<v0::Constant>(data_type, ov::Shape{config.indexer_head_dim}, std::vector<float>{1.0f});
        const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
        const auto one = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        const auto activation_head_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        const auto state_head_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
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
                                 v0::Constant::create(ov::element::i32, ov::Shape{1}, {config.indexer_rotary_dim})},
                0));
            return rg.make<v3::Broadcast>(rope, target);
        };

        const auto key_length_i32 = key_length.get_element_type() == ov::element::i32
                                        ? batched_key_length
                                        : rg.make<v0::Convert>(batched_key_length, ov::element::i32);

        // Without padding, QSAIndexer follows the QSAPrepare cache contract and takes the cache as is.
        // With padding, QSAIndexer keeps the cache shape, while QSAPrepare output 0 holds the complete blocks of the
        // longest sequence, so the cache is grown by zero blocks before the call.
        ov::Output<ov::Node> block_keys_cache = rg.make<v0::Unsqueeze>(past_block_keys, state_head_axis);
        ov::Output<ov::Node> needed_blocks;
        if (padding) {
            const auto max_key_length = rg.make<v1::ReduceMax>(key_length_i32, zero, true);
            needed_blocks = rg.make<v1::Divide>(max_key_length,
                                                v0::Constant::create(ov::element::i32, ov::Shape{1}, {key_block_size}));
            const auto past_block_count = dims_of(past_block_keys, {1});
            const auto missing_blocks = rg.make<v1::Subtract>(needed_blocks, past_block_count);
            const auto cache_growth_blocks = rg.make<v1::Maximum>(missing_blocks, zero);
            const auto cache_growth_shape = rg.add(ov::op::util::make_try_fold<v0::Concat>(
                ov::OutputVector{dims_of(raw_keys, {0}),
                                 one,
                                 cache_growth_blocks,
                                 v0::Constant::create(ov::element::i32, ov::Shape{1}, {config.indexer_head_dim})},
                0));
            const auto zero_key = rg.make<v0::Constant>(data_type, ov::Shape{}, std::vector<float>{0.0f});
            const auto cache_padding = rg.make<v3::Broadcast>(zero_key, cache_growth_shape);
            constexpr int64_t state_blocks_axis = 2;
            block_keys_cache =
                rg.make<v0::Concat>(ov::OutputVector{block_keys_cache, cache_padding}, state_blocks_axis);
        }

        const auto query_blhd =
            rg.make<v1::Transpose>(query, v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
        const auto fused = rg.make<op_internal::QSAIndexer>(ov::OutputVector{query_blhd,
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
        ov::Output<ov::Node> block_keys = squeeze_state(QSAIndexerOutputIdx::BLOCK_INDEX_KEYS_CACHE, 0);
        if (padding) {
            // QSAIndexer returns the grown cache on BLOCK_INDEX_KEYS_CACHE, while QSAPrepare output 0 holds only the
            // complete blocks of the longest sequence, so the replacement is sliced back to that block count.
            // A Constant key_length folds the count so the sliced shape stays as static as the original output.
            ov::Output<ov::Node> block_count = needed_blocks;
            if (const auto lengths = ov::as_type_ptr<v0::Constant>(key_length.get_node_shared_ptr())) {
                const auto values = lengths->cast_vector<int64_t>();
                if (!values.empty()) {
                    const auto max_length = *std::max_element(values.begin(), values.end());
                    block_count = v0::Constant::create(ov::element::i32, ov::Shape{1}, {max_length / key_block_size});
                }
            }
            block_keys.get_node()->set_friendly_name(prepare->get_friendly_name() + ".0.capacity");
            // Expose only the required block range, not any extra capacity added before QSAIndexer.
            const auto sliced = rg.make<v8::Slice>(block_keys, zero, block_count, one, one);
            sliced->set_friendly_name(prepare->get_friendly_name() + ".0");
            block_keys = sliced;
        }
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

template <class Indexer>
FuseQSAIndexerQueryRoPE<Indexer>::FuseQSAIndexerQueryRoPE() {
    MATCHER_SCOPE(FuseQSAIndexerQueryRoPE);
    using namespace ov::pass::pattern;
    using In = typename Indexer::InputIdx;

    const auto p_query = any_input();
    const auto p_rope_cos = any_input();
    const auto p_rope_sin = any_input();
    // NopElimination can replace the head-axis Unsqueeze of [batch, length, D] with a Reshape.
    const auto with_head_axis = [](const ov::Output<ov::Node>& rows) {
        const auto unsqueeze = wrap_type<v0::Unsqueeze>({rows, wrap_type<v0::Constant>(value_matches("1"))});
        const auto reshape = wrap_type<v1::Reshape>({rows, wrap_const()}, [](const ov::Output<ov::Node>& output) {
            const auto& shape = output.get_node()->get_input_partial_shape(0);
            return shape.rank() == 3 && shape[1].is_static() && shape[1].get_length() > 0 && shape[2].is_static() &&
                   shape[2].get_length() > 0 &&
                   output.get_partial_shape() == ov::PartialShape{shape[0], 1, shape[1], shape[2]};
        });
        return unsqueeze | reshape;
    };
    const auto p_head_first = optional<v1::Reshape>({p_query, any_input()});
    const auto p_rope = wrap_type<op_internal::RoPE>(
        {p_head_first, with_head_axis(p_rope_cos), with_head_axis(p_rope_sin)},
        [](const ov::Output<ov::Node>& output) {
            const auto rope = ov::as_type<op_internal::RoPE>(output.get_node());
            auto config = rope->get_config();
            config.input_trans0213 = false;
            const auto& cos_shape = rope->get_input_partial_shape(1);
            return op_internal::is_plain_rotate_half(config) && cos_shape.rank().is_static() &&
                   cos_shape[-1] == static_cast<int64_t>(config.rotary_ndims);
        },
        {{"config.input_interleaved", false}, {"config.use_rope_cache", false}});
    const auto p_token_major = optional<v1::Transpose>({p_rope, wrap_type<v0::Constant>(value_matches("0, 2, 1, 3"))});
    const auto p_query_reshape = optional<v1::Reshape>({p_token_major, any_input()}, rank_equals(2));
    // FuseQSAIndexer broadcasts the cos / sin rows with a v3 Broadcast, which ConvertBroadcast3 may turn into a
    // numpy v1 Broadcast before this pass.
    const auto rope_rows = [](const ov::Output<ov::Node>& rows) {
        const auto batch_rows = optional<v3::Broadcast>({rows, any_input()}) |
                                wrap_type<v1::Broadcast>({rows, any_input(), any_input()}, {{"mode", "numpy"}});
        return optional<v1::Reshape>({batch_rows, any_input()}, rank_equals(2));
    };
    const auto p_cos_reshape = rope_rows(p_rope_cos);
    const auto p_sin_reshape = rope_rows(p_rope_sin);
    OutputVector pattern_inputs(In::COUNT);
    for (auto& input : pattern_inputs) {
        input = any_input();
    }
    pattern_inputs[In::INDEX_QUERY] = p_query_reshape;
    pattern_inputs[In::ROPE_COS] = p_cos_reshape;
    pattern_inputs[In::ROPE_SIN] = p_sin_reshape;
    const auto p_indexer = wrap_type<Indexer>(pattern_inputs, {{"query_rope", false}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto indexer = ov::as_type_ptr<Indexer>(m.get_match_root());
        OPENVINO_ASSERT(indexer, "FuseQSAIndexerQueryRoPE: expected QSA indexer");
        if (transformation_callback(indexer)) {
            return false;
        }
        const auto rope = ov::as_type_ptr<op_internal::RoPE>(pattern_map.at(p_rope).get_node_shared_ptr());
        OPENVINO_ASSERT(rope, "FuseQSAIndexerQueryRoPE: expected RoPE");
        const bool head_first = pattern_map.count(p_head_first) != 0;
        const bool transposed = pattern_map.count(p_token_major) != 0;
        if (rope->get_config().input_trans0213 != transposed ||
            (!transposed && (!head_first || !pattern_map.count(p_query_reshape)))) {
            return false;
        }
        const auto query = transposed ? rope->input_value(0) : pattern_map.at(p_query);
        auto config = indexer->get_config();
        config.query_rope = true;
        auto inputs = indexer->input_values();
        NodeRegistry rg;
        NodeVector sources{rope};
        if (head_first && !transposed) {
            sources.push_back(pattern_map.at(p_head_first).get_node_shared_ptr());
        }
        if (transposed) {
            sources.push_back(pattern_map.at(p_token_major).get_node_shared_ptr());
        }
        if (pattern_map.count(p_query_reshape)) {
            const auto query_reshape = pattern_map.at(p_query_reshape).get_node_shared_ptr();
            inputs[In::INDEX_QUERY] =
                rg.add(query_reshape->clone_with_new_inputs({query, query_reshape->input_value(1)}));
            sources.push_back(query_reshape);
        } else {
            inputs[In::INDEX_QUERY] = query;
        }
        const auto fused = rg.make<Indexer>(inputs, config);
        fused->set_friendly_name(indexer->get_friendly_name());
        sources.push_back(indexer);
        ov::copy_runtime_info(sources, rg.get());
        ov::replace_node(indexer, fused);
        register_new_node(fused);
        return true;
    };
    register_matcher(std::make_shared<Matcher>(p_indexer, matcher_name), callback);
}

template <class Indexer>
FuseQSAIndexerQueryNorm<Indexer>::FuseQSAIndexerQueryNorm() {
    MATCHER_SCOPE(FuseQSAIndexerQueryNorm);
    using namespace ov::pass::pattern;
    using In = typename Indexer::InputIdx;

    const auto keeps_precision = rank_more_than(0) && [](const ov::Output<ov::Node>& output) {
        return output.get_element_type() == output.get_node()->get_input_element_type(0);
    };
    const auto gamma_matches = [](const ov::Output<ov::Node>& output) {
        // Predicates run before inputs are matched, so guard the gamma access.
        const auto rms = output.get_node();
        if (rms->get_input_size() != 2 || rms->get_input_element_type(1) != output.get_element_type()) {
            return false;
        }
        const auto& gamma_shape = rms->get_input_partial_shape(1);
        // Gamma can be flattened to [D] only when its leading dimensions are singleton.
        return gamma_shape.rank().is_static() && gamma_shape.size() > 0 &&
               gamma_shape[-1] == output.get_partial_shape()[-1] &&
               std::all_of(gamma_shape.begin(), gamma_shape.end() - 1, [](const Dimension& dim) {
                   return dim == 1;
               });
    };
    const auto p_gamma = any_input();
    const auto p_rms_affine = wrap_type<op_internal::RMS>({any_input(), p_gamma},
                                                          keeps_precision && gamma_matches,
                                                          {{"axis", -1}, {"elementwise_affine", true}});
    const auto p_rms_non_affine =
        wrap_type<op_internal::RMS>({any_input()}, keeps_precision, {{"axis", -1}, {"elementwise_affine", false}});
    const auto p_rms = p_rms_affine | p_rms_non_affine;
    const auto p_query_reshape = optional<v1::Reshape>({p_rms, any_input()}, rank_equals(2));
    OutputVector pattern_inputs(In::COUNT);
    for (auto& input : pattern_inputs) {
        input = any_input();
    }
    pattern_inputs[In::INDEX_QUERY] = p_query_reshape;
    const auto p_indexer = wrap_type<Indexer>(pattern_inputs, {{"query_norm", false}});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto indexer = ov::as_type_ptr<Indexer>(m.get_match_root());
        OPENVINO_ASSERT(indexer, "FuseQSAIndexerQueryNorm: expected QSA indexer");
        auto config = indexer->get_config();
        const auto rms = ov::as_type_ptr<op_internal::RMS>(pattern_map.at(p_rms).get_node_shared_ptr());
        OPENVINO_ASSERT(rms, "FuseQSAIndexerQueryNorm: expected RMS");
        // The indexer normalizes each head separately; an RMS spanning several heads is not equivalent.
        const auto head_dim = Dimension(static_cast<int64_t>(config.indexer_head_dim));
        if (rms->get_epsilon() != config.epsilon || rms->get_output_partial_shape(0)[-1] != head_dim ||
            transformation_callback(indexer)) {
            return false;
        }
        config.query_norm = true;
        auto inputs = indexer->input_values();
        NodeRegistry rg;
        NodeVector sources{rms, indexer};
        if (pattern_map.count(p_query_reshape)) {
            const auto query_reshape = pattern_map.at(p_query_reshape).get_node_shared_ptr();
            inputs[In::INDEX_QUERY] =
                rg.add(query_reshape->clone_with_new_inputs({rms->input_value(0), query_reshape->input_value(1)}));
            sources.push_back(query_reshape);
        } else {
            inputs[In::INDEX_QUERY] = rms->input_value(0);
        }
        if (pattern_map.count(p_gamma)) {
            auto gamma = pattern_map.at(p_gamma);
            if (gamma.get_partial_shape().size() > 1) {
                const auto shape =
                    rg.make<v0::Constant>(element::i64,
                                          Shape{1},
                                          std::vector<int64_t>{static_cast<int64_t>(config.indexer_head_dim)});
                gamma = rg.add(ov::op::util::make_try_fold<v1::Reshape>(gamma, shape, false));
            }
            inputs[In::Q_NORM_WEIGHT] = gamma;
        } else {
            // The non-affine RMS has no learned scaling, so use unit weights.
            inputs[In::Q_NORM_WEIGHT] = rg.make<v0::Constant>(inputs[In::Q_NORM_WEIGHT].get_element_type(),
                                                              Shape{config.indexer_head_dim},
                                                              std::vector<float>{1.0f});
        }
        const auto fused = rg.make<Indexer>(inputs, config);
        fused->set_friendly_name(indexer->get_friendly_name());
        ov::copy_runtime_info(sources, rg.get());
        ov::replace_node(indexer, fused);
        register_new_node(fused);
        return true;
    };
    register_matcher(std::make_shared<Matcher>(p_indexer, matcher_name), callback);
}

template class FuseQSAIndexerQueryRoPE<op_internal::QSAIndexer>;
template class FuseQSAIndexerQueryRoPE<op_internal::PagedQSAIndexer>;
template class FuseQSAIndexerQueryNorm<op_internal::QSAIndexer>;
template class FuseQSAIndexerQueryNorm<op_internal::PagedQSAIndexer>;

bool FuseQSAIndexerQuery::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(FuseQSAIndexerQuery);
    // QSAIndexer normalizes before it rotates, so RoPE is absorbed over the whole model before RMS.
    GraphRewrite rope;
    rope.set_pass_config(get_pass_config());
    rope.add_matcher<FuseQSAIndexerQueryRoPE<op_internal::QSAIndexer>>();
    rope.add_matcher<FuseQSAIndexerQueryRoPE<op_internal::PagedQSAIndexer>>();
    GraphRewrite norm;
    norm.set_pass_config(get_pass_config());
    norm.add_matcher<FuseQSAIndexerQueryNorm<op_internal::QSAIndexer>>();
    norm.add_matcher<FuseQSAIndexerQueryNorm<op_internal::PagedQSAIndexer>>();
    const bool rope_fused = rope.run_on_model(model);
    return norm.run_on_model(model) || rope_fused;
}

}  // namespace ov::pass
