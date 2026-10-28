// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_attention_fusion.hpp"

#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "intel_gpu/op/paged_qsa_indexer.hpp"
#include "intel_gpu/op/qsa_indexer.hpp"
#include "intel_gpu/op/sparse_pa.hpp"
#include "intel_gpu/op/sparse_sdpa.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/paged_sparse_attention.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/sparse_attention.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"

namespace ov::intel_gpu {

namespace {

bool static_dim(const ov::PartialShape& ps, size_t axis, size_t& value) {
    if (ps.rank().is_dynamic() || axis >= ps.size() || ps[axis].is_dynamic() || ps[axis].get_length() < 0) {
        return false;
    }
    value = static_cast<size_t>(ps[axis].get_length());
    return true;
}

bool has_rank(const ov::Output<ov::Node>& out, size_t rank) {
    const auto& ps = out.get_partial_shape();
    return ps.rank().is_static() && ps.size() == rank;
}

/// Collects the nodes a rewrite creates so runtime info can be copied onto all of them.
struct NodeBuilder {
    ov::NodeVector nodes;

    template <class T, class... Args>
    std::shared_ptr<T> add(Args&&... args) {
        auto node = std::make_shared<T>(std::forward<Args>(args)...);
        nodes.push_back(node);
        return node;
    }

    ov::Output<ov::Node> i64(const std::vector<int64_t>& values) {
        auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{values.size()}, values);
        nodes.push_back(c);
        return c;
    }

    /// [1]-shaped i64 extent of `axis` of `x`.
    ov::Output<ov::Node> dim(const ov::Output<ov::Node>& x, int64_t axis) {
        auto shape = add<ov::op::v3::ShapeOf>(x, ov::element::i64);
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        nodes.push_back(axis_const);
        return add<ov::op::v8::Gather>(shape, i64({axis}), axis_const);
    }

    /// `key_length` ([N] or [1], i32/i64) as the i32 [N] tensor the kernels read.
    ov::Output<ov::Node> key_length_i32(const ov::Output<ov::Node>& key_length, const ov::Output<ov::Node>& batch_ref) {
        ov::Output<ov::Node> kl = key_length;
        if (kl.get_element_type() != ov::element::i32) {
            kl = add<ov::op::v0::Convert>(kl, ov::element::i32);
        }
        return add<ov::op::v3::Broadcast>(kl, dim(batch_ref, 0));
    }

    ov::Output<ov::Node> to_i32(const ov::Output<ov::Node>& x) {
        return x.get_element_type() == ov::element::i32 ? x : ov::Output<ov::Node>(add<ov::op::v0::Convert>(x, ov::element::i32));
    }
};

ov::Output<ov::Node> strip_converts(ov::Output<ov::Node> out) {
    while (ov::is_type<ov::op::v0::Convert>(out.get_node())) {
        out = out.get_node()->input_value(0);
    }
    return out;
}

/// Looks through the ops that only change the layout of RoPE cos/sin.
ov::Output<ov::Node> strip_layout_ops(ov::Output<ov::Node> out) {
    while (ov::is_type<ov::op::v0::Convert>(out.get_node()) || ov::is_type<ov::op::v0::Unsqueeze>(out.get_node()) ||
           ov::is_type<ov::op::v0::Squeeze>(out.get_node()) || ov::is_type<ov::op::v1::Reshape>(out.get_node())) {
        out = out.get_node()->input_value(0);
    }
    return out;
}

bool const_values(const ov::Output<ov::Node>& out, std::vector<int64_t>& values) {
    auto c = ov::util::get_constant_from_source(out);
    if (!c) {
        return false;
    }
    values = c->cast_vector<int64_t>();
    return true;
}

bool is_last_axis(int64_t axis, size_t rank) {
    return axis == -1 || axis == static_cast<int64_t>(rank) - 1;
}

/// `out` = Slice(src, start:stop, step 1) along the last axis of a rank-4 tensor.
bool match_last_axis_slice(const ov::Output<ov::Node>& out, int64_t& start, int64_t& stop, ov::Output<ov::Node>& src) {
    auto slice = ov::as_type_ptr<ov::op::v8::Slice>(strip_converts(out).get_node_shared_ptr());
    if (!slice || slice->get_input_size() < 5) {
        return false;
    }
    std::vector<int64_t> begin, end, step, axes;
    if (!const_values(slice->input_value(1), begin) || !const_values(slice->input_value(2), end) || !const_values(slice->input_value(3), step) ||
        !const_values(slice->input_value(4), axes) || begin.size() != 1 || end.size() != 1 || step != std::vector<int64_t>{1} || axes.size() != 1 ||
        !is_last_axis(axes[0], 4)) {
        return false;
    }
    start = begin[0];
    stop = end[0];
    src = slice->input_value(0);
    return true;
}

/// The index-query side of the public pair, still in front of SparseAttentionIndexer as regular
/// graph ops: RMSNorm(gain) -> RoPE (partial, rotate-half) -> optional [N, L, H, D] -> [N, H, L, D]
/// Transpose. QSAIndexer applies both itself, so it takes the raw query and the gain.
struct QuerySide {
    ov::Output<ov::Node> raw_blhd;               ///< RMSNorm input as [N, L, Hi, Di]
    std::shared_ptr<ov::op::v0::Constant> gain;  ///< [Di], f16
    double epsilon = 0.0;
};

bool match_query_side(const ov::Output<ov::Node>& index_query,
                      const ov::Output<ov::Node>& rope_cos,
                      const ov::Output<ov::Node>& rope_sin,
                      size_t hi,
                      size_t di,
                      size_t dr,
                      NodeBuilder& b,
                      QuerySide& q) {
    ov::Output<ov::Node> cur = strip_converts(index_query);
    bool blhd = false;
    if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(cur.get_node_shared_ptr())) {
        std::vector<int64_t> order;
        if (!const_values(transpose->input_value(1), order) || order != std::vector<int64_t>{0, 2, 1, 3}) {
            return false;
        }
        cur = strip_converts(transpose->input_value(0));
        blhd = true;
    }

    // Partial rotary: Concat(RoPE(x[..., :Dr]), x[..., Dr:]); full-width rotary: RoPE(x).
    ov::Output<ov::Node> x;
    std::shared_ptr<ov::op::internal::RoPE> rope;
    if (auto concat = ov::as_type_ptr<ov::op::v0::Concat>(cur.get_node_shared_ptr())) {
        if (concat->get_input_size() != 2 || !is_last_axis(concat->get_axis(), 4)) {
            return false;
        }
        rope = ov::as_type_ptr<ov::op::internal::RoPE>(strip_converts(concat->input_value(0)).get_node_shared_ptr());
        int64_t rotated_begin = 0, rotated_end = 0, passed_begin = 0, passed_end = 0;
        ov::Output<ov::Node> rotated_src, passed_src;
        if (!rope || !match_last_axis_slice(rope->input_value(0), rotated_begin, rotated_end, rotated_src) ||
            !match_last_axis_slice(concat->input_value(1), passed_begin, passed_end, passed_src) || rotated_src != passed_src || rotated_begin != 0 ||
            rotated_end != static_cast<int64_t>(dr) || passed_begin != static_cast<int64_t>(dr) || passed_end < static_cast<int64_t>(di)) {
            return false;
        }
        x = rotated_src;
    } else {
        rope = ov::as_type_ptr<ov::op::internal::RoPE>(cur.get_node_shared_ptr());
        if (!rope) {
            return false;
        }
        x = rope->input_value(0);
    }
    if (rope->get_input_size() != 3 || !ov::op::internal::is_plain_rotate_half(rope->get_config()) || rope->get_config().rotary_ndims != dr) {
        return false;
    }
    // The query rotates with the very cos/sin the block keys are rotated with.
    if (strip_layout_ops(rope->input_value(1)) != strip_layout_ops(rope_cos) || strip_layout_ops(rope->input_value(2)) != strip_layout_ops(rope_sin)) {
        return false;
    }

    // RMSNorm with gain: RMS(x, gain) or RMS(x) * gain.
    ov::Output<ov::Node> norm = strip_converts(x);
    ov::Output<ov::Node> gain;
    std::shared_ptr<ov::op::internal::RMS> rms;
    if (auto fused = ov::as_type_ptr<ov::op::internal::RMS>(norm.get_node_shared_ptr()); fused && fused->get_input_size() == 2) {
        rms = fused;
        gain = fused->input_value(1);
    } else if (auto mul = ov::as_type_ptr<ov::op::v1::Multiply>(norm.get_node_shared_ptr())) {
        for (size_t i = 0; i < 2 && !rms; i++) {
            auto candidate = ov::as_type_ptr<ov::op::internal::RMS>(strip_converts(mul->input_value(i)).get_node_shared_ptr());
            if (candidate && candidate->get_input_size() == 1) {
                rms = candidate;
                gain = mul->input_value(1 - i);
            }
        }
    }
    if (!rms || !is_last_axis(rms->get_axis(), 4)) {
        return false;
    }
    auto gain_const = ov::util::get_constant_from_source(gain);
    if (!gain_const || ov::shape_size(gain_const->get_shape()) != di) {
        return false;
    }

    // The norm input has the layout of the RoPE output; the kernels read [N, L, H, D].
    ov::Output<ov::Node> raw = rms->input_value(0);
    size_t heads = 0, width = 0;
    if (!has_rank(raw, 4) || raw.get_element_type() != ov::element::f16 || !static_dim(raw.get_partial_shape(), blhd ? 2 : 1, heads) ||
        !static_dim(raw.get_partial_shape(), 3, width) || heads != hi || width != di) {
        return false;
    }
    if (!blhd) {
        raw = b.add<ov::op::v1::Transpose>(raw, b.i64({0, 2, 1, 3}));
    }
    q.raw_blhd = raw;
    q.gain = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{di}, gain_const->cast_vector<float>());
    b.nodes.push_back(q.gain);
    q.epsilon = rms->get_epsilon();
    return true;
}

// ---------------------------------------------------------------------------------------------
// Qwen Sparse Attention, non-paged: opset17 QSAPrepare + SparseAttentionIndexer -> QSAIndexer
// ---------------------------------------------------------------------------------------------
bool fuse_qsa_indexer(const std::shared_ptr<ov::op::v17::SparseAttentionIndexer>& indexer) {
    using Prep = ov::op::v17::QSAPrepare;
    using QSAIn = ov::intel_gpu::op::QSAIndexer::InputIdx;
    using QSAOut = ov::intel_gpu::op::QSAIndexer::OutputIdx;

    // index_weights (DSA) have no kernel support.
    if (indexer->get_input_size() != 3 || indexer->get_num_selection_heads() != 1) {
        return false;
    }
    if (indexer->input_value(1).get_index() != 0) {
        return false;
    }
    auto prepare = ov::as_type_ptr<Prep>(indexer->get_input_node_shared_ptr(1));
    if (!prepare || prepare->get_input_size() != 8) {
        return false;
    }
    const int64_t r = prepare->get_key_block_size();
    const int64_t token_budget = indexer->get_token_budget();
    if (r < 2 || indexer->get_key_block_size() != r || token_budget <= 0 || token_budget % r != 0) {
        return false;
    }

    const auto index_query = indexer->input_value(0);
    const auto raw_keys = prepare->input_value(0);
    const auto past_block_keys = prepare->input_value(1);
    const auto key_length = prepare->input_value(2);
    const auto rope_cos = prepare->input_value(3);
    const auto rope_sin = prepare->input_value(4);
    const auto k_norm_weight = prepare->input_value(5);
    const auto past_raw = prepare->input_value(6);
    const auto past_rope = prepare->input_value(7);

    // Both ops are fed the same lengths in a real layer; the kernel takes a single key_length.
    if (indexer->input_value(2) != key_length) {
        return false;
    }
    // Every optional input must be present: the kernels always normalize and rotate.
    for (const auto& optional : {rope_cos, rope_sin, k_norm_weight, past_rope}) {
        if (ov::util::is_empty_constant_tensor(optional)) {
            return false;
        }
    }
    if (!has_rank(index_query, 4) || !has_rank(raw_keys, 3) || !has_rank(past_block_keys, 3) || !has_rank(rope_cos, 3) || !has_rank(rope_sin, 3) ||
        !has_rank(past_raw, 3) || !has_rank(past_rope, 2)) {
        return false;
    }
    size_t hi = 0, di = 0, dr = 0, di_keys = 0;
    if (!static_dim(index_query.get_partial_shape(), 1, hi) || !static_dim(index_query.get_partial_shape(), 3, di) ||
        !static_dim(raw_keys.get_partial_shape(), 2, di_keys) || !static_dim(rope_cos.get_partial_shape(), 2, dr) || di != di_keys || dr == 0) {
        return false;
    }
    for (const auto& data : {index_query, raw_keys, past_block_keys, rope_cos, rope_sin, k_norm_weight, past_raw, past_rope}) {
        if (data.get_element_type() != ov::element::f16) {
            return false;
        }
    }

    ov::intel_gpu::op::QSAIndexer::Config config;
    config.key_block_size = static_cast<size_t>(r);
    config.block_topk = static_cast<size_t>(token_budget / r);
    config.num_selection_heads = 1;
    config.indexer_rotary_dim = dr;
    config.indexer_head_dim = di;
    config.epsilon = prepare->get_epsilon();
    config.scale = 1.0f / std::sqrt(static_cast<float>(di));
    std::string reason;
    if (!ov::intel_gpu::op::QSAIndexer::validate_config(config, reason)) {
        return false;
    }

    NodeBuilder b;
    QuerySide query;
    if (!match_query_side(index_query, rope_cos, rope_sin, hi, di, dr, b, query) || std::abs(query.epsilon - config.epsilon) > 1e-4 * config.epsilon) {
        return false;
    }
    auto axis_const = [&](int64_t axis) {
        auto c = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {axis});
        b.nodes.push_back(c);
        return c;
    };

    // The kernels take [B, L, H, D] activations; the public ops are heads-first / head-less.
    auto raw_keys_blhd = b.add<ov::op::v0::Unsqueeze>(raw_keys, axis_const(2));
    auto past_raw_bhrd = b.add<ov::op::v0::Unsqueeze>(past_raw, axis_const(1));
    ov::Output<ov::Node> key_length_b = b.key_length_i32(key_length, raw_keys);

    // RoPE inputs broadcast over the batch in the public ops.
    auto rope_for_batch = [&](const ov::Output<ov::Node>& rope) -> ov::Output<ov::Node> {
        const auto& rope_ps = rope.get_partial_shape();
        const auto& key_ps = raw_keys.get_partial_shape();
        const bool rope_single = rope_ps[0].is_static() && rope_ps[0].get_length() == 1;
        const bool keys_single = key_ps[0].is_static() && key_ps[0].get_length() == 1;
        if (!rope_single || keys_single) {
            return rope;
        }
        auto target = b.add<ov::op::v0::Concat>(ov::OutputVector{b.dim(raw_keys, 0), b.dim(raw_keys, 1), b.i64({static_cast<int64_t>(dr)})}, 0);
        return b.add<ov::op::v3::Broadcast>(rope, target);
    };

    // The block-key state grows to the complete blocks of the longest sequence before the call.
    auto key_length_i64 = b.add<ov::op::v0::Convert>(key_length, ov::element::i64);
    auto longest = b.add<ov::op::v1::ReduceMax>(key_length_i64, axis_const(0), true);
    auto needed_blocks = b.add<ov::op::v1::Divide>(longest, b.i64({r}));
    auto missing = b.add<ov::op::v1::Subtract>(needed_blocks, b.dim(past_block_keys, 1));
    auto growth = b.add<ov::op::v1::Maximum>(missing, b.i64({0}));
    auto zero_f16 = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.0f});
    b.nodes.push_back(zero_f16);
    auto growth_shape = b.add<ov::op::v0::Concat>(ov::OutputVector{b.dim(raw_keys, 0), b.i64({1}), growth, b.i64({static_cast<int64_t>(di)})}, 0);
    auto growth_zeros = b.add<ov::op::v3::Broadcast>(zero_f16, growth_shape);
    auto past_block_keys_bhbd = b.add<ov::op::v0::Unsqueeze>(past_block_keys, axis_const(1));
    auto cache_in = b.add<ov::op::v0::Concat>(ov::OutputVector{past_block_keys_bhbd, growth_zeros}, 2);

    ov::OutputVector args(QSAIn::COUNT);
    args[QSAIn::INDEX_QUERY] = query.raw_blhd;
    args[QSAIn::RAW_INDEX_KEYS] = raw_keys_blhd;
    args[QSAIn::ROPE_COS] = rope_for_batch(rope_cos);
    args[QSAIn::ROPE_SIN] = rope_for_batch(rope_sin);
    args[QSAIn::KEY_LENGTH] = key_length_b;
    args[QSAIn::Q_NORM_WEIGHT] = query.gain;
    args[QSAIn::K_NORM_WEIGHT] = k_norm_weight;
    args[QSAIn::BLOCK_INDEX_KEYS_CACHE] = cache_in;
    args[QSAIn::PAST_INCOMPLETE_RAW_K] = past_raw_bhrd;
    args[QSAIn::PAST_INCOMPLETE_ROPE] = past_rope;
    auto qsa_indexer = b.add<ov::intel_gpu::op::QSAIndexer>(args, config);

    // The public QSAPrepare outputs are the new values of its three states.
    auto block_keys_out = b.add<ov::op::v0::Squeeze>(qsa_indexer->output(QSAOut::BLOCK_INDEX_KEYS_CACHE), axis_const(1));
    auto raw_out = b.add<ov::op::v0::Squeeze>(qsa_indexer->output(QSAOut::PAST_INCOMPLETE_RAW_K), axis_const(1));

    ov::copy_runtime_info(ov::NodeVector{prepare, indexer}, b.nodes);
    ov::replace_node(prepare, ov::OutputVector{block_keys_out->output(0), raw_out->output(0), qsa_indexer->output(QSAOut::PAST_INCOMPLETE_ROPE)});
    ov::replace_node(indexer, ov::OutputVector{qsa_indexer->output(QSAOut::SELECTED_BLOCK_INDICES), qsa_indexer->output(QSAOut::SELECTED_BLOCK_COUNT)});
    qsa_indexer->set_friendly_name(indexer->get_friendly_name());
    return true;
}

// ---------------------------------------------------------------------------------------------
// Qwen Sparse Attention, non-paged: opset17 SparseAttention -> SparseSDPA
// ---------------------------------------------------------------------------------------------
bool convert_sparse_attention(const std::shared_ptr<ov::op::v17::SparseAttention>& attention) {
    using In = ov::intel_gpu::op::SparseSDPA::InputIdx;

    const size_t input_count = attention->get_input_size();
    // The GPU kernels have no attention-sink, non-causal or query-block support.
    if (input_count > 7 || !attention->get_causal() || attention->get_query_block_size() != 1) {
        return false;
    }

    const auto query = attention->input_value(0);
    const auto key = attention->input_value(1);
    const auto value = attention->input_value(2);
    const auto block_indices = attention->input_value(3);
    const auto block_count = attention->input_value(4);
    if (!has_rank(query, 4) || !has_rank(key, 4) || !has_rank(value, 4) || !has_rank(block_indices, 4)) {
        return false;
    }
    for (const auto& data : {query, key, value}) {
        if (data.get_element_type() != ov::element::f16) {
            return false;
        }
    }

    ov::intel_gpu::op::SparseSDPA::Config config;
    if (!static_dim(query.get_partial_shape(), 1, config.num_heads) || !static_dim(query.get_partial_shape(), 3, config.k_head_size) ||
        !static_dim(key.get_partial_shape(), 1, config.num_kv_heads) || !static_dim(value.get_partial_shape(), 3, config.v_head_size) ||
        !static_dim(block_indices.get_partial_shape(), 1, config.num_selection_heads)) {
        return false;
    }
    config.key_block_size = static_cast<size_t>(attention->get_key_block_size());
    config.query_block_size = 1;
    config.causal = true;
    std::string reason;
    if (config.num_selection_heads != 1 || !ov::intel_gpu::op::SparseSDPA::validate_config(config, reason)) {
        return false;
    }

    // The scale is folded into the primitive, so it has to be constant.
    std::shared_ptr<ov::op::v0::Constant> scale;
    if (input_count >= 7) {
        scale = ov::util::get_constant_from_source(attention->input_value(6));
        if (!scale || ov::shape_size(scale->get_shape()) != 1) {
            return false;
        }
    }

    NodeBuilder b;
    // Without key_length there is no padding: every key position is valid.
    ov::Output<ov::Node> key_length_b;
    if (input_count >= 6) {
        key_length_b = b.key_length_i32(attention->input_value(5), query);
    } else {
        auto positions = b.dim(key, 2);
        auto positions_i32 = b.add<ov::op::v0::Convert>(positions, ov::element::i32);
        key_length_b = b.add<ov::op::v3::Broadcast>(positions_i32, b.dim(query, 0));
    }

    ov::OutputVector args(In::COUNT_MIN);
    args[In::QUERY] = query;
    args[In::KEY] = key;
    args[In::VALUE] = value;
    args[In::SELECTED_BLOCK_INDICES] = b.to_i32(block_indices);
    args[In::SELECTED_BLOCK_COUNT] = b.to_i32(block_count);
    args[In::KEY_LENGTH] = key_length_b;
    if (scale) {
        args.push_back(scale);
    }
    auto sparse_sdpa = b.add<ov::intel_gpu::op::SparseSDPA>(args, config, attention->get_output_element_type(0));
    sparse_sdpa->set_friendly_name(attention->get_friendly_name());
    ov::copy_runtime_info(attention, b.nodes);
    ov::replace_node(attention, sparse_sdpa);
    return true;
}

// ---------------------------------------------------------------------------------------------
// Qwen Sparse Attention, paged: core PagedQSAIndexer / PagedSparseAttention -> gpu_opset ops
// ---------------------------------------------------------------------------------------------
void tag_in_place_cache(const ov::Output<ov::Node>& cache) {
    if (auto param = ov::as_type_ptr<ov::op::v0::Parameter>(cache.get_node_shared_ptr())) {
        param->output(0).get_rt_info()[in_place_cache_rt_key] = true;
    }
}

bool convert_paged_qsa_indexer(const std::shared_ptr<ov::op::internal::PagedQSAIndexer>& indexer) {
    using In = ov::op::internal::PagedQSAIndexer::InputIdx;
    if (indexer->get_input_size() != In::COUNT) {
        return false;
    }
    auto gpu_indexer = std::make_shared<ov::intel_gpu::op::PagedQSAIndexer>(indexer->input_values(), indexer->get_config());
    // The three caches are updated by the device in place.
    for (size_t idx : {In::INDEX_KEY_CACHE, In::INDEX_BLOCK_ROPE_CACHE, In::BLOCK_INDEX_KEYS_CACHE}) {
        tag_in_place_cache(indexer->input_value(idx));
    }
    gpu_indexer->set_friendly_name(indexer->get_friendly_name());
    ov::copy_runtime_info(indexer, gpu_indexer);
    ov::replace_node(indexer, gpu_indexer);
    return true;
}

bool convert_paged_sparse_attention(const std::shared_ptr<ov::op::internal::PagedSparseAttention>& attention) {
    using In = ov::intel_gpu::op::SparsePA::InputIdx;
    if (attention->get_input_size() != In::COUNT) {
        return false;
    }

    // The kernels are specialized for the head geometry, which the op only carries in its shapes.
    const auto& query_ps = attention->get_input_partial_shape(In::QUERY);
    const auto& key_cache_ps = attention->get_input_partial_shape(In::KEY_CACHE);
    const auto& value_cache_ps = attention->get_input_partial_shape(In::VALUE_CACHE);
    const auto& selected_ps = attention->get_input_partial_shape(In::SELECTED_BLOCK_INDICES);
    if (query_ps.rank() != 2 || key_cache_ps.rank() != 4 || value_cache_ps.rank() != 4 || selected_ps.rank() != 3) {
        return false;
    }
    size_t query_width = 0;
    ov::intel_gpu::op::SparsePA::Config config;
    if (!static_dim(query_ps, 1, query_width) || !static_dim(key_cache_ps, 1, config.num_kv_heads) || !static_dim(key_cache_ps, 3, config.k_head_size) ||
        !static_dim(value_cache_ps, 3, config.v_head_size) || !static_dim(selected_ps, 1, config.num_selection_heads) || config.k_head_size == 0) {
        return false;
    }
    config.num_heads = query_width / config.k_head_size;
    // The core op no longer ties the projection widths to the caches, so the conversion has to.
    size_t key_width = 0, value_width = 0;
    if (!static_dim(attention->get_input_partial_shape(In::KEY), 1, key_width) || !static_dim(attention->get_input_partial_shape(In::VALUE), 1, value_width) ||
        query_width % config.k_head_size != 0 || key_width != config.num_kv_heads * config.k_head_size ||
        value_width != config.num_kv_heads * config.v_head_size) {
        return false;
    }
    config.key_block_size = static_cast<size_t>(attention->get_key_block_size());
    config.query_block_size = 1;
    config.causal = true;
    std::string reason;
    if (!ov::intel_gpu::op::SparsePA::validate_config(config, reason)) {
        return false;
    }

    auto sparse_pa = std::make_shared<ov::intel_gpu::op::SparsePA>(attention->input_values(), config);
    sparse_pa->set_friendly_name(attention->get_friendly_name());
    ov::copy_runtime_info(attention, sparse_pa);
    ov::replace_node(attention, sparse_pa);
    return true;
}

}  // namespace

QSAIndexerFusion::QSAIndexerFusion() {
    using namespace ov::pass::pattern;

    auto indexer_m = wrap_type<ov::op::v17::SparseAttentionIndexer>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        return fuse_qsa_indexer(ov::as_type_ptr<ov::op::v17::SparseAttentionIndexer>(m.get_match_root()));
    };
    register_matcher(std::make_shared<Matcher>(indexer_m, "QSAIndexerFusion"), callback);
}

SparseSDPAFusion::SparseSDPAFusion() {
    using namespace ov::pass::pattern;

    auto attention_m = wrap_type<ov::op::v17::SparseAttention>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }
        return convert_sparse_attention(ov::as_type_ptr<ov::op::v17::SparseAttention>(m.get_match_root()));
    };
    register_matcher(std::make_shared<Matcher>(attention_m, "SparseSDPAFusion"), callback);
}

PagedQSAIndexerConversion::PagedQSAIndexerConversion() {
    using namespace ov::pass::pattern;

    auto indexer_m = wrap_type<ov::op::internal::PagedQSAIndexer>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto root = m.get_match_root();
        // wrap_type also matches the derived gpu_opset op this pass produces.
        if (ov::is_type<ov::intel_gpu::op::PagedQSAIndexer>(root) || transformation_callback(root)) {
            return false;
        }
        return convert_paged_qsa_indexer(ov::as_type_ptr<ov::op::internal::PagedQSAIndexer>(root));
    };
    register_matcher(std::make_shared<Matcher>(indexer_m, "PagedQSAIndexerConversion"), callback);
}

PagedSparseAttentionConversion::PagedSparseAttentionConversion() {
    using namespace ov::pass::pattern;

    auto attention_m = wrap_type<ov::op::internal::PagedSparseAttention>();

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto root = m.get_match_root();
        // wrap_type also matches the derived gpu_opset op this pass produces.
        if (ov::is_type<ov::intel_gpu::op::SparsePA>(root) || transformation_callback(root)) {
            return false;
        }
        return convert_paged_sparse_attention(ov::as_type_ptr<ov::op::internal::PagedSparseAttention>(root));
    };
    register_matcher(std::make_shared<Matcher>(attention_m, "PagedSparseAttentionConversion"), callback);
}

SparseAttentionFusion::SparseAttentionFusion() {
    add_matcher<QSAIndexerFusion>();
    add_matcher<SparseSDPAFusion>();
    add_matcher<PagedQSAIndexerConversion>();
    add_matcher<PagedSparseAttentionConversion>();
}

}  // namespace ov::intel_gpu
