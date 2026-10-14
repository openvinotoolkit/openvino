// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <memory>
#include <numeric>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/decompositions/low_precision_dequantize.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/bitwise_and.hpp"
#include "openvino/op/bitwise_left_shift.hpp"
#include "openvino/op/bitwise_or.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/logical_and.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using ov::pass::pattern::Matcher;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v3 = ov::op::v3;
namespace v4 = ov::op::v4;
namespace v5 = ov::op::v5;
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;
namespace v15 = ov::op::v15;

namespace {
// Finite lowest() of the attention compute type: the masked-logit fill (a fully-masked row must not softmax to NaN).
float finite_lowest(const ov::element::Type& type) {
    if (type == ov::element::f16)
        return static_cast<float>(std::numeric_limits<ov::float16>::lowest());
    if (type == ov::element::bf16)
        return static_cast<float>(std::numeric_limits<ov::bfloat16>::lowest());
    return std::numeric_limits<float>::lowest();
}
}  // namespace
ov::pass::GroupQueryAttentionDecomposition::GroupQueryAttentionDecomposition() {
    MATCHER_SCOPE(GroupQueryAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::internal::GroupQueryAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node = ov::as_type_ptr<ov::op::internal::GroupQueryAttention>(
            pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }

        auto new_output_node = decompose(node);
        ov::replace_node(node, new_output_node);
        return true;
    };

    auto m = std::make_shared<Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}

ov::OutputVector ov::pass::GroupQueryAttentionDecomposition::decompose(
    std::shared_ptr<ov::op::internal::GroupQueryAttention> node) {
    using GQAInputs = ov::op::internal::GroupQueryAttentionInputs;

    const auto num_heads = node->get_num_heads();
    const auto kv_num_heads = node->get_kv_num_heads();
    const auto scale = node->get_scale();
    const auto do_rotary = node->get_do_rotary();
    const auto rotary_interleaved = node->get_rotary_interleaved();
    const auto local_window_size = node->get_local_window_size();
    const auto smooth_softmax = node->get_smooth_softmax();
    const auto causal = node->get_causal();
    const auto softcap = node->get_softcap();
    const auto qk_output = node->get_qk_output();
    // softcap and qk_output need the explicit attention chain instead of SDPA.
    const bool use_explicit_attention =
        softcap > 0.0f || qk_output != ov::op::internal::GroupQueryAttentionQKOutput::NONE;

    const auto has_input = [&](const GQAInputs input_pos) {
        const auto pos = static_cast<size_t>(input_pos);
        return (pos < node->get_input_size()) && !ov::util::is_empty_constant_tensor(node->input_value(pos));
    };

    const auto get_input = [&](const GQAInputs input_pos) -> ov::Output<ov::Node> {
        const auto original_pos = static_cast<size_t>(input_pos);
        const bool exists = has_input(input_pos);
        OPENVINO_ASSERT(exists, "Missing required GroupQueryAttention input at original position ", original_pos);
        return node->input_value(original_pos);
    };

    auto Q = get_input(GQAInputs::QUERY);
    auto K = get_input(GQAInputs::KEY);
    auto V = get_input(GQAInputs::VALUE);
    auto past_key = get_input(GQAInputs::PAST_KEY);
    auto past_value = get_input(GQAInputs::PAST_VALUE);
    auto seqlens_k = get_input(GQAInputs::SEQLENS_K);

    // Quantized KV cache (com.microsoft spec): past/present KV are i8/u8/f8e4m3 and are dequantized before the
    // attention math and (re)quantized when appended to the cache. Scales live at ONNX K_SCALE / V_SCALE positions.
    const bool kv_quantized = node->is_kv_quantized();
    const auto kv_cache_bit_width = node->get_kv_cache_bit_width();
    const auto k_quant_type = node->get_k_quant_type();
    const auto v_quant_type = node->get_v_quant_type();
    const auto kv_cache_type = past_key.get_element_type();
    ov::Output<ov::Node> k_scale, v_scale;

    // Get k_scale and v_scale from their actual input indices.
    // Note: validate_and_infer_types() already verified these indices are valid when kv_quantized is true,
    // so we skip redundant bounds checks here.
    if (kv_quantized) {
        k_scale = get_input(GQAInputs::K_SCALE);
        v_scale = get_input(GQAInputs::V_SCALE);
    }

    // The length of all tokens (past + current) is `seqlens_k` + 1.
    // current = Q.shape[2], past = `seqlens_k` + 1 - current

    const auto T = Q.get_element_type();
    const auto q_shape = register_new_node<v3::ShapeOf>(Q);
    const auto current_seqlen = get_dimensions(q_shape, {2});
    const auto head_size_node = get_dimensions(q_shape, {3});

    const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    const auto zero_without_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    const auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    const auto one_without_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    const auto two = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    const bool shared_kv = node->is_shared_kv();
    // A separate key/value with a dynamic length may resolve to 0 at runtime (ORT shared KV), so its own length
    // drives the cache arithmetic: the past keeps total - S_kv rows while the causal offset stays total - S_q.
    const bool independent_kv_len = node->has_independent_kv_length();
    const auto kv_seqlen = independent_kv_len ? get_dimensions(register_new_node<v3::ShapeOf>(K), {2}) : nullptr;
    const auto seqlens_elemi64 = register_new_node<v0::Convert>(seqlens_k, ov::element::i64);
    const auto real_seqlens = register_new_node<v1::Add>(seqlens_elemi64, one);

    // Per-batch lengths (ORT: one seqlens_k entry per batch) whenever the batch is not statically 1. A static batch
    // of 1 keeps the scalar-length graph below unchanged, since plugins and the NPU compiler pattern-match it; the
    // windowed cache stays single-batch.
    const auto& q_ps = node->get_input_partial_shape(static_cast<size_t>(GQAInputs::QUERY));
    const bool batched = !node->get_sliding_window_cache() &&
                         !(q_ps.rank().is_static() && q_ps[0].is_static() && q_ps[0].get_length() == 1);
    // [B, 1, 1]: total = seqlens_k + 1, the causal offset of the first query (ORT: 0 for a first, right-padded
    // prompt, total - S otherwise), and the row the first new key is written at.
    std::shared_ptr<ov::Node> total_b, q_past_b, kv_past_b;
    if (batched) {
        // [B, 1, 1] with B taken from the query, so a seqlens_k without one entry per batch fails the Reshape
        // instead of broadcasting one length over every batch (ORT requires batch_size elements).
        const auto per_batch = register_new_node<v0::Concat>(
            ov::OutputVector{get_dimensions(q_shape, {0}),
                             register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1}))},
            0);
        total_b = register_new_node<v1::Reshape>(real_seqlens, per_batch, false);
        q_past_b = register_new_node<v1::Maximum>(register_new_node<v1::Subtract>(total_b, current_seqlen), zero);
        kv_past_b = independent_kv_len
                        ? register_new_node<v1::Maximum>(register_new_node<v1::Subtract>(total_b, kv_seqlen), zero)
                        : q_past_b;
    }
    // A single batch entry uses scalar ([1]) lengths; only built when not batched (the [1] reshape needs batch 1).
    std::shared_ptr<ov::Node> seqlens_1d, past_seqlen;
    if (!batched) {
        seqlens_1d = register_new_node<v1::Reshape>(real_seqlens, one, false);
        past_seqlen = register_new_node<v1::Subtract>(seqlens_1d, current_seqlen);
    }
    const auto curr_seqlen_scalar = register_new_node<v0::Squeeze>(current_seqlen);

    // QK-Norm is applied to the new Q/K heads before RoPE (ORT); shared KV has no new keys.
    if (node->has_qk_norm()) {
        const auto epsilon = node->get_qk_norm_epsilon();
        Q = make_qk_norm(Q, get_input(GQAInputs::Q_NORM_WEIGHT), epsilon);
        if (!shared_kv) {
            K = make_qk_norm(K, get_input(GQAInputs::K_NORM_WEIGHT), epsilon);
        }
    }

    if (do_rotary) {
        // Get cos_cache and sin_cache from their actual input indices (ONNX COS_CACHE and SIN_CACHE).
        // validate_and_infer_types() already verified these inputs exist and indices are valid when do_rotary is true.
        auto cos_cache = get_input(GQAInputs::COS_CACHE);
        auto sin_cache = get_input(GQAInputs::SIN_CACHE);

        ov::Output<ov::Node> position_ids =
            register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
        // Batched: [B, S] positions (past_b + s, or the given [B, S] position_ids); cos/sin become [B, S, D/2].
        const auto pos_axis = batched ? one : zero;
        if (batched) {
            const auto minus_one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
            const auto rows = register_new_node<v0::Concat>(ov::OutputVector{minus_one, current_seqlen}, 0);
            if (has_input(GQAInputs::POSITION_IDS)) {
                position_ids = register_new_node<v1::Reshape>(get_input(GQAInputs::POSITION_IDS), rows, false);
            } else {
                const auto column = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{2}, {-1, 1}));
                position_ids =
                    register_new_node<v1::Add>(register_new_node<v1::Reshape>(q_past_b, column, false), position_ids);
            }
        } else if (has_input(GQAInputs::POSITION_IDS)) {
            // Flatten position_ids to 1D so that Gather produces 2D [seqlen, head_size/2] output,
            // ensuring correct 4D shapes after Unsqueeze in rotaryEmbedding.
            const auto neg_one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
            position_ids = register_new_node<v1::Reshape>(get_input(GQAInputs::POSITION_IDS), neg_one, false);
        } else {
            position_ids = register_new_node<v1::Add>(position_ids, past_seqlen);
        }

        const auto cos = register_new_node<v8::Gather>(cos_cache, position_ids, zero);
        const auto sin = register_new_node<v8::Gather>(sin_cache, position_ids, zero);
        Q = rotaryEmbedding(Q, cos, sin, rotary_interleaved);
        // Shared KV has no new keys: the past keys were already rotated when they were appended.
        if (independent_kv_len) {
            // The new keys sit at the same positions as the queries; take only the first S_kv rows (all of them
            // unless S_kv resolves to 0 at runtime).
            const auto cos_k = register_new_node<v8::Slice>(cos, zero, kv_seqlen, one, pos_axis);
            const auto sin_k = register_new_node<v8::Slice>(sin, zero, kv_seqlen, one, pos_axis);
            K = rotaryEmbedding(K, cos_k, sin_k, rotary_interleaved);
        } else if (!shared_kv) {
            K = rotaryEmbedding(K, cos, sin, rotary_interleaved);
        }
    }
    // A preallocated cache (static capacity and step length, whatever the batch) is written in place. A
    // zero-capacity past (an absent ONNX past or an empty cache) has nothing to write into, so it grows by
    // concatenation instead. Same rule as the op's present shape inference.
    const auto is_static_input = node->has_fixed_kv_capacity();

    // Quantize-on-write: when the cache is quantized, quantize the (post-RoPE) current K/V into the cache type
    // before appending them, so the assembled present cache stays quantized and the past bytes are preserved
    // verbatim (no re-rounding of past tokens). Matches ONNX Runtime MLAS/CUDA semantics.
    if (kv_quantized && !shared_kv) {
        K = quantize_kv(K, k_scale, kv_num_heads, kv_cache_bit_width, k_quant_type, kv_cache_type);
        V = quantize_kv(V, v_scale, kv_num_heads, kv_cache_bit_width, v_quant_type, kv_cache_type);
    }

    // past_seqlen expressed in the coordinate system the attention mask uses. Equals the absolute past
    // length for a full-length cache; a windowed cache overrides it with the resident row count.
    ov::Output<ov::Node> mask_past_seqlen = batched ? ov::Output<ov::Node>() : past_seqlen->output(0);
    // Absolute key position of the KV buffer's first slot, used to align an external attention_bias (indexed
    // by absolute key). 0 for a full-length cache (slot j == absolute key j); a windowed cache rolls, so its
    // first slot holds absolute key P - resident_rows (set in the windowed branches below).
    ov::Output<ov::Node> bias_col_offset = zero;
    ov::Output<ov::Node> present_k, present_v;

    if (shared_kv) {
        // Shared KV (ORT kv_sequence_length == 0): attend to the past buffer as is (zero copy) and return it
        // unchanged as present. Rows at or beyond `total` are hidden by the explicit mask (causal offset
        // total - S_q via mask_past_seqlen), so no Slice of the cache is needed.
        K = past_key;
        V = past_value;
        present_k = past_key;
        present_v = past_value;
    } else if (past_key.get_partial_shape()[2].is_static() && past_key.get_partial_shape()[2].get_length() == 0) {
        // With no past tokens, the current K/V already are the complete present cache.
        present_k = K;
        present_v = V;
    } else if (node->get_sliding_window_cache()) {
        // Windowed KV cache (capacity C, rolled with front eviction). end_before/end_after are the resident
        // row counts before/after appending the S new tokens (see windowed_cache_end).
        const auto capacity = get_dimensions(register_new_node<v3::ShapeOf>(past_key), {2});
        const auto capacity_scalar = register_new_node<v0::Squeeze>(capacity);
        const auto abs_past_scalar = register_new_node<v0::Squeeze>(past_seqlen);  // P
        const auto abs_total_scalar = register_new_node<v0::Squeeze>(seqlens_1d);  // P + S
        const auto end_before = windowed_cache_end(abs_past_scalar, capacity_scalar, local_window_size);
        const auto end_after = windowed_cache_end(abs_total_scalar, capacity_scalar, local_window_size);

        // Static single-token decode (S == 1) always fits the window and uses the in-place Gather +
        // ScatterUpdate assembly (static-shape friendly). Otherwise (dynamic S) a multi-token step may cross
        // an eviction, making the in-place kept = end_after - S negative, so it takes the staging path below.
        // A statically-known S > 1 is rejected up front (FE + op), so it never reaches here.
        const auto& q_ps = node->get_input_partial_shape(0);
        const bool static_single_token = q_ps.rank().is_static() && q_ps.rank().get_length() == 4 &&
                                         q_ps[2].is_static() && q_ps[2].get_length() == 1;

        const auto scatter_axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
        const auto zeros =
            register_new_node<v3::Broadcast>(register_new_node(v0::Constant::create(kv_cache_type, ov::Shape{}, {0})),
                                             register_new_node<v3::ShapeOf>(past_key));

        if (static_single_token) {
            // present = [survivors, new, zeros] left-aligned in the C buffer: the last kept = end_after - S
            // resident rows, then the S new tokens.
            const auto kept = register_new_node<v1::Subtract>(end_after, curr_seqlen_scalar);  // end_after - S
            const auto survivor_start = register_new_node<v1::Subtract>(end_before, kept);
            const auto kept_row =
                register_new_node<v4::Range>(zero_without_shape, kept, one_without_shape, ov::element::i64);
            const auto survivor_idx = register_new_node<v1::Add>(kept_row, survivor_start);
            const auto survivor_k = register_new_node<v8::Gather>(past_key, survivor_idx, two);
            const auto survivor_v = register_new_node<v8::Gather>(past_value, survivor_idx, two);
            const auto kept_idx = kept_row;
            const auto new_row = register_new_node<v4::Range>(zero_without_shape,
                                                              curr_seqlen_scalar,
                                                              one_without_shape,
                                                              ov::element::i64);
            const auto new_idx = register_new_node<v1::Add>(new_row, kept);

            present_k = register_new_node<v3::ScatterUpdate>(zeros, kept_idx, survivor_k, scatter_axis);
            present_k = register_new_node<v3::ScatterUpdate>(present_k, new_idx, K, scatter_axis);
            present_v = register_new_node<v3::ScatterUpdate>(zeros, kept_idx, survivor_v, scatter_axis);
            present_v = register_new_node<v3::ScatterUpdate>(present_v, new_idx, V, scatter_axis);

            // Attention runs on the trimmed [survivors, new] buffer (end_after rows, no trailing padding), so
            // the kernel's native LOWER_RIGHT + window sees the newest token as the last row. present_k/present_v
            // (left-aligned, padded to capacity C) stay the ORT-compatible op outputs. Mirrors the staging path.
            K = register_new_node<v0::Concat>(ov::OutputVector{survivor_k, K}, 2);
            V = register_new_node<v0::Concat>(ov::OutputVector{survivor_v, V}, 2);
            mask_past_seqlen = register_new_node<v0::Unsqueeze>(kept, zero);
            // First resident slot holds absolute key P - kept (the survivors start there).
            bias_col_offset =
                register_new_node<v0::Unsqueeze>(register_new_node<v1::Subtract>(abs_past_scalar, kept), zero);
        } else {
            // Staging (ORT parity): attend against a temp buffer of the end_before resident rows + S new
            // tokens, then write only the surviving tail (last end_after rows) back into the capacity-C cache.
            const auto end_before_1d = register_new_node<v0::Unsqueeze>(end_before, zero);
            const auto resident_k = register_new_node<v8::Slice>(past_key, zero, end_before_1d, one, two);
            const auto resident_v = register_new_node<v8::Slice>(past_value, zero, end_before_1d, one, two);
            const auto temp_k = register_new_node<v0::Concat>(ov::OutputVector{resident_k, K}, 2);
            const auto temp_v = register_new_node<v0::Concat>(ov::OutputVector{resident_v, V}, 2);

            // tail = last end_after rows of the temp buffer, scattered into [0, end_after) of the C buffer.
            const auto temp_len = register_new_node<v1::Add>(end_before, curr_seqlen_scalar);
            const auto tail_start = register_new_node<v1::Subtract>(temp_len, end_after);
            const auto tail_start_1d = register_new_node<v0::Unsqueeze>(tail_start, zero);
            const auto temp_len_1d = register_new_node<v0::Unsqueeze>(temp_len, zero);
            const auto tail_k = register_new_node<v8::Slice>(temp_k, tail_start_1d, temp_len_1d, one, two);
            const auto tail_v = register_new_node<v8::Slice>(temp_v, tail_start_1d, temp_len_1d, one, two);
            const auto present_row =
                register_new_node<v4::Range>(zero_without_shape, end_after, one_without_shape, ov::element::i64);
            present_k = register_new_node<v3::ScatterUpdate>(zeros, present_row, tail_k, scatter_axis);
            present_v = register_new_node<v3::ScatterUpdate>(zeros, present_row, tail_v, scatter_axis);

            // Attention runs on the temp buffer; only the returned present is the capacity-C tail.
            K = temp_k;
            V = temp_v;
            mask_past_seqlen = register_new_node<v0::Unsqueeze>(end_before, zero);
            // Temp buffer's first slot holds absolute key P - end_before.
            bias_col_offset =
                register_new_node<v0::Unsqueeze>(register_new_node<v1::Subtract>(abs_past_scalar, end_before), zero);
        }
    } else if (batched) {
        // Per-batch cache write (ORT ConcatStateChunkGQA, left-aligned): batch b keeps its first kv_past_b rows and
        // gets its S_kv new rows at [kv_past_b, kv_past_b + S_kv). The cache is viewed as [B * Nkv * L, D] rows and
        // written with one ScatterUpdate(axis 0) at row (b * Nkv + h) * L + kv_past_b + s: in place on a static buffer,
        // after a single Concat(past, new) for a growing cache (L = past rows + S_kv; rows past a batch's total are
        // unspecified, as in ORT). Index arithmetic stays rank 4 ([B, Nkv, S_kv, 1]) so plugins broadcast it in their
        // native layout.
        const auto kv_rows = independent_kv_len ? kv_seqlen : current_seqlen;
        const auto as_shape = [&](std::initializer_list<int64_t> shape) {
            return register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{shape.size()}, shape));
        };
        const auto batch = register_new_node<v0::Squeeze>(get_dimensions(register_new_node<v3::ShapeOf>(K), {0}));
        const auto batch_idx = register_new_node<v1::Reshape>(
            register_new_node<v4::Range>(zero_without_shape, batch, one_without_shape, ov::element::i64),
            as_shape({-1, 1, 1, 1}),
            false);
        std::vector<int64_t> heads(static_cast<size_t>(kv_num_heads));
        std::iota(heads.begin(), heads.end(), int64_t{0});
        const auto head_idx = register_new_node(
            v0::Constant::create(ov::element::i64, ov::Shape{1, static_cast<size_t>(kv_num_heads), 1, 1}, heads));
        const auto bh_idx = register_new_node<v1::Add>(
            register_new_node<v1::Multiply>(
                batch_idx,
                register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {kv_num_heads}))),
            head_idx);  // [B, Nkv, 1, 1]
        const auto new_row = register_new_node<v1::Add>(
            register_new_node<v1::Reshape>(kv_past_b, as_shape({-1, 1, 1, 1}), false),
            register_new_node<v1::Reshape>(register_new_node<v4::Range>(zero_without_shape,
                                                                        register_new_node<v0::Squeeze>(kv_rows),
                                                                        one_without_shape,
                                                                        ov::element::i64),
                                           as_shape({1, 1, -1, 1}),
                                           false));  // [B, 1, S_kv, 1]
        const auto write = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
            const ov::Output<ov::Node> buffer =
                is_static_input ? past : register_new_node<v0::Concat>(ov::OutputVector{past, current}, 2)->output(0);
            const auto buffer_shape = register_new_node<v3::ShapeOf>(buffer);
            const auto rows_per_head = get_dimensions(buffer_shape, {2});
            const auto flat_idx = register_new_node<v1::Reshape>(
                register_new_node<v1::Add>(register_new_node<v1::Multiply>(bh_idx, rows_per_head), new_row),
                as_shape({-1}),
                false);  // [B * Nkv * S_kv]
            const auto row_shape =
                register_new_node<v0::Concat>(ov::OutputVector{as_shape({-1}), get_dimensions(buffer_shape, {3})}, 0);
            const auto rows =
                register_new_node<v3::ScatterUpdate>(register_new_node<v1::Reshape>(buffer, row_shape, false),
                                                     flat_idx,
                                                     register_new_node<v1::Reshape>(current, row_shape, false),
                                                     zero);
            return register_new_node<v1::Reshape>(rows, buffer_shape, false);
        };
        K = write(past_key, K);
        V = write(past_value, V);
        present_k = K;
        present_v = V;
    } else if (is_static_input) {
        // Static full-length cache (max length, valid KVs left-aligned). Insert current K/V at
        // [past_seqlen, past_seqlen + curr_seqlen] with ScatterUpdate, keeping the buffer shape.
        // An out-of-range past_seqlen is ScatterUpdate's own bounds-check responsibility, not something to
        // guard against here via a graph-level clamp; the decomposition assumes the caller-supplied
        // seqlens_k stays within the declared cache capacity.
        std::shared_ptr<ov::Node> scatter_idx =
            register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
        scatter_idx = register_new_node<v1::Add>(scatter_idx, past_seqlen);
        const auto scatter_axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
        K = register_new_node<v3::ScatterUpdate>(past_key, scatter_idx, K, scatter_axis);
        V = register_new_node<v3::ScatterUpdate>(past_value, scatter_idx, V, scatter_axis);
        present_k = K;
        present_v = V;
    } else {
        auto construct_kv_cache = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
            return register_new_node<v0::Concat>(ov::OutputVector{past, current}, 2);
        };
        const ov::Output<ov::Node> kept_past = independent_kv_len
                                                   ? register_new_node<v1::Subtract>(seqlens_1d, kv_seqlen)->output(0)
                                                   : past_seqlen->output(0);
        past_key = register_new_node<v8::Slice>(past_key, zero, kept_past, one, two);
        past_value = register_new_node<v8::Slice>(past_value, zero, kept_past, one, two);
        K = construct_kv_cache(past_key, K);
        V = construct_kv_cache(past_value, V);
        present_k = K;
        present_v = V;
    }

    // The explicit attention chain consumes plain (dequantized) K/V, so skip the compressed-KV SDPA form.
    std::optional<CompressedKV> compressed_kv;
    if (!use_explicit_attention) {
        compressed_kv = prepare_compressed_kv(node, K, V, k_scale, v_scale);
    }

    // Dequantize the assembled cache to the compute (float) type for the attention math. Everything downstream
    // (head broadcast, mask, SDPA) then operates in float exactly as in the non-quantized path.
    if (kv_quantized && !compressed_kv) {
        K = dequantize_kv(K, k_scale, kv_num_heads, kv_cache_bit_width, k_quant_type, T);
        V = dequantize_kv(V, v_scale, kv_num_heads, kv_cache_bit_width, v_quant_type, T);
    }

    const auto concat_kv_len = get_dimensions(register_new_node<v3::ShapeOf>(K), {2});
    const auto concat_kv_len_scalar = register_new_node<v0::Squeeze>(concat_kv_len);

    // Broadcast KV if grouped query attention
    const size_t kv_num_heads_factor = num_heads / kv_num_heads;
    // The explicit chain groups query heads per KV head itself (no KV replication), see make_explicit_attention.
    if (kv_num_heads_factor > 1 && !compressed_kv && !use_explicit_attention) {
        const auto kv_shape = register_new_node<v3::ShapeOf>(K);
        const auto kv_shape_prev_2 = get_dimensions(kv_shape, {0, 1});
        const auto kv_shape_last_2 = get_dimensions(kv_shape, {2, 3});
        auto new_kv_shape = register_new_node<v0::Concat>(ov::NodeVector{kv_shape_prev_2, one, kv_shape_last_2}, 0);
        K = register_new_node<v1::Reshape>(K, new_kv_shape, false);
        V = register_new_node<v1::Reshape>(V, new_kv_shape, false);
        K = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, K), 2);
        V = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, V), 2);
        const auto q_shape = register_new_node<v3::ShapeOf>(Q);
        const auto q_shape_prev_2 = get_dimensions(q_shape, {0, 1});
        auto extended_kv_shape = register_new_node<v0::Concat>(ov::NodeVector{q_shape_prev_2, kv_shape_last_2}, 0);
        K = register_new_node<v1::Reshape>(K, extended_kv_shape, false);
        V = register_new_node<v1::Reshape>(V, extended_kv_shape, false);
    }

    ov::Output<ov::Node> external_bias;
    if (has_input(GQAInputs::ATTENTION_BIAS)) {
        external_bias = get_input(GQAInputs::ATTENTION_BIAS);
    }
    const bool has_head_sink = has_input(GQAInputs::HEAD_SINK);
    const bool has_sink = has_head_sink || smooth_softmax;
    // The explicit chain has no is_causal fallback, and shared KV must hide the past rows beyond `total`, so both
    // always take the explicit (base) mask; overrides may otherwise drop it in favor of SDPA's causal flag.
    const bool explicit_mask = use_explicit_attention || shared_kv;
    // qk_output before softmax reports masked positions as 0 (ORT); they are found on the bias-free mask, so a large
    // negative attention_bias reads logit + bias. The mask is then built once without the bias, and the aligned bias
    // is added on top (same op count as the fused form).
    const bool split_bias =
        qk_output == ov::op::internal::GroupQueryAttentionQKOutput::BEFORE_SOFTMAX && external_bias.get_node();
    const auto mask_bias = split_bias ? ov::Output<ov::Node>() : external_bias;
    std::shared_ptr<ov::Node> mask =
        batched         ? make_batched_attention_mask(curr_seqlen_scalar,
                                              concat_kv_len_scalar,
                                              concat_kv_len,
                                              q_past_b,
                                              total_b,
                                              T,
                                              causal,
                                              local_window_size,
                                              mask_bias)
        : explicit_mask ? GroupQueryAttentionDecomposition::make_attention_mask(curr_seqlen_scalar,
                                                                                concat_kv_len_scalar,
                                                                                concat_kv_len,
                                                                                mask_past_seqlen,
                                                                                T,
                                                                                causal,
                                                                                local_window_size,
                                                                                mask_bias,
                                                                                bias_col_offset,
                                                                                node->get_sliding_window_cache(),
                                                                                scale,
                                                                                has_sink)
                        : make_attention_mask(curr_seqlen_scalar,
                                              concat_kv_len_scalar,
                                              concat_kv_len,
                                              mask_past_seqlen,
                                              T,
                                              causal,
                                              local_window_size,
                                              mask_bias,
                                              bias_col_offset,
                                              node->get_sliding_window_cache(),
                                              scale,
                                              has_sink);

    // qk_output before softmax reports masked positions as 0 (ORT). They are found on the mask without
    // attention_bias, so a large negative bias value is reported as logit + bias, not as masked.
    ov::Output<ov::Node> unbiased_mask = mask;
    if (split_bias) {
        mask = register_new_node<v1::Add>(
            mask,
            align_attention_bias(external_bias, batched ? zero : bias_col_offset, concat_kv_len, T, batched));
    }

    // head_sink (input 11) or smooth_softmax add an extra logit to the softmax denominator. SDPA models
    // this with its sink input: a [1, num_heads, 1, 1] tensor appended as one logit column, included in
    // the softmax, then sliced out. head_sink provides a per-head value; plain smooth_softmax uses 0.
    ov::Output<ov::Node> sink;
    if (has_sink) {
        const auto sink_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, -1, 1, 1}));
        if (has_head_sink) {
            auto head_sink = get_input(GQAInputs::HEAD_SINK);
            if (head_sink.get_element_type() != T) {
                head_sink = register_new_node<v0::Convert>(head_sink, T);
            }
            sink = register_new_node<v1::Reshape>(head_sink, sink_shape, false);
        } else {
            const auto num_heads_1d =
                register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads}));
            sink = register_new_node<v3::Broadcast>(
                register_new_node(v0::Constant::create(T, ov::Shape{}, {0})),
                register_new_node<v0::Concat>(ov::NodeVector{one, num_heads_1d, one, one}, 0));
        }
    }

    // Explicit scale: the op scale or the default 1/sqrt(head_size).
    const auto make_scale_node = [&]() -> ov::Output<ov::Node> {
        if (scale != 0.0f) {
            return register_new_node(v0::Constant::create(T, Shape{}, {scale}));
        }
        const auto head_size_t = register_new_node<v0::Convert>(head_size_node, T);
        const auto neg_half = register_new_node(v0::Constant::create(T, Shape{}, {-0.5f}));
        return register_new_node<v0::Squeeze>(register_new_node<ov::op::v1::Power>(head_size_t, neg_half));
    };

    ov::Output<ov::Node> qga_output;
    ov::Output<ov::Node> qk_scores;
    if (use_explicit_attention) {
        const auto attention = make_explicit_attention(Q,
                                                       K,
                                                       V,
                                                       mask,
                                                       unbiased_mask,
                                                       make_scale_node(),
                                                       sink,
                                                       softcap,
                                                       kv_num_heads,
                                                       qk_output);
        qga_output = attention[0];
        if (qk_output != ov::op::internal::GroupQueryAttentionQKOutput::NONE) {
            // ONNX Runtime lays output_qk out over the total_sequence_length input, which a static or shared cache
            // does not tie to seqlens_k; the per-batch lengths only mask. Extra cache rows are dropped here.
            const auto three = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}));
            const auto total_cols = register_new_node<v1::Reshape>(
                register_new_node<v0::Convert>(get_input(GQAInputs::TOTAL_SEQUENCE_LENGTH), ov::element::i64),
                one,
                false);
            qk_scores = register_new_node<v8::Slice>(attention[1], zero, total_cols, one, three);
        }
    } else if (sink.get_node_shared_ptr()) {
        // SDPA's 6-input form requires an explicit scale.
        qga_output = make_sdpa(Q, K, V, mask, make_scale_node(), sink, false, local_window_size, compressed_kv);
    } else if (scale != 0.0f) {
        auto scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
        qga_output = make_sdpa(Q, K, V, mask, scale_node, {}, false, local_window_size, compressed_kv);
    } else {
        qga_output = make_sdpa(Q, K, V, mask, {}, {}, !mask, local_window_size, compressed_kv);
    }

    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads * head_size)
    auto perm = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
    auto qga_output_transposed = register_new_node<v1::Transpose>(qga_output, perm);
    auto dim_merge_shape = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1}));
    auto output = register_new_node<v1::Reshape>(qga_output_transposed, dim_merge_shape, true)->output(0);

    if (qk_scores.get_node()) {
        return {output, present_k, present_v, qk_scores};
    }
    return {output, present_k, present_v};
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::make_sdpa(
    const ov::Output<ov::Node>& query,
    const ov::Output<ov::Node>& key,
    const ov::Output<ov::Node>& value,
    const ov::Output<ov::Node>& mask,
    const ov::Output<ov::Node>& scale,
    const ov::Output<ov::Node>& sink,
    bool is_causal,
    [[maybe_unused]] int64_t local_window_size,
    [[maybe_unused]] const std::optional<CompressedKV>& compressed_kv) {
    // local_window_size and compressed_kv are ignored by the reference v13::ScaledDotProductAttention op;
    // plugin overrides may apply them when the underlying kernel supports SWA / compressed KV natively.
    if (sink.get_node()) {
        return register_new_node<v13::ScaledDotProductAttention>(query, key, value, mask, scale, sink, is_causal);
    }
    if (scale.get_node()) {
        return register_new_node<v13::ScaledDotProductAttention>(query, key, value, mask, scale, is_causal);
    }
    if (mask.get_node()) {
        return register_new_node<v13::ScaledDotProductAttention>(query, key, value, mask, is_causal);
    }
    return register_new_node<v13::ScaledDotProductAttention>(query, key, value, is_causal);
}

ov::OutputVector ov::pass::GroupQueryAttentionDecomposition::make_explicit_attention(
    const ov::Output<ov::Node>& query,
    const ov::Output<ov::Node>& key,
    const ov::Output<ov::Node>& value,
    const ov::Output<ov::Node>& mask,
    const ov::Output<ov::Node>& unbiased_mask,
    const ov::Output<ov::Node>& scale,
    const ov::Output<ov::Node>& sink,
    float softcap,
    int64_t kv_num_heads,
    ov::op::internal::GroupQueryAttentionQKOutput qk_output) {
    using QKOutput = ov::op::internal::GroupQueryAttentionQKOutput;
    // SDPA has no hook between Q*K^T and softmax, so this uses the explicit chain in ONNX Runtime order
    // (gqa_attention_base.h): scale -> [softcap * tanh(x / softcap)] -> + bias/mask -> softmax (+ sink) -> * V.
    const auto T = query.get_element_type();
    // Grouped-query attention without replicating K/V: view Q [B, N, S, D] as [B, Nkv, G, S, D] (query head
    // h = kv * G + g, matching ORT) and let MatMul broadcast it against K/V viewed as [B, Nkv, 1, L, D]. The
    // reshapes are free views of contiguous tensors; scores are taken back to [B, N, S, L] for the mask/softmax.
    const auto q_shape = register_new_node<v3::ShapeOf>(query);
    const auto k_shape = register_new_node<v3::ShapeOf>(key);
    const auto batch = get_dimensions(q_shape, {0});
    const auto num_heads = get_dimensions(q_shape, {1});
    const auto q_len_head = get_dimensions(q_shape, {2, 3});
    const auto kv_len = get_dimensions(k_shape, {2});
    const auto kv_heads = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {kv_num_heads}));
    const auto minus_one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    const auto axis_2 = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    const auto grouped_q_shape =
        register_new_node<v0::Concat>(ov::NodeVector{batch, kv_heads, minus_one, q_len_head}, 0);
    const auto grouped_q = register_new_node<v1::Reshape>(query, grouped_q_shape, false);
    const auto grouped_k = register_new_node<v0::Unsqueeze>(key, axis_2);
    const auto grouped_v = register_new_node<v0::Unsqueeze>(value, axis_2);
    const auto grouped_scores = register_new_node<v0::MatMul>(grouped_q, grouped_k, false, true);
    const auto scores_shape =
        register_new_node<v0::Concat>(ov::NodeVector{batch, num_heads, get_dimensions(q_shape, {2}), kv_len}, 0);
    const auto scores = register_new_node<v1::Reshape>(grouped_scores, scores_shape, false);
    std::shared_ptr<ov::Node> logits;
    if (softcap > 0.0f) {
        const auto inv_cap = register_new_node(v0::Constant::create(T, Shape{}, {1.0f / softcap}));
        const auto cap = register_new_node(v0::Constant::create(T, Shape{}, {softcap}));
        const auto scaled = register_new_node<v1::Multiply>(scores, register_new_node<v1::Multiply>(scale, inv_cap));
        logits = register_new_node<v1::Multiply>(register_new_node<v0::Tanh>(scaled), cap);
    } else {
        logits = register_new_node<v1::Multiply>(scores, scale);
    }
    logits = register_new_node<v1::Add>(logits, mask);
    if (T != ov::element::f32) {
        // The mask fills with the type's finite lowest(); keep lowest() + logit from overflowing to -inf in
        // reduced precision, so a fully-masked row cannot softmax to NaN.
        const auto lowest = T == ov::element::f16 ? static_cast<float>(std::numeric_limits<ov::float16>::lowest())
                                                  : static_cast<float>(std::numeric_limits<ov::bfloat16>::lowest());
        logits = register_new_node<v1::Maximum>(logits, register_new_node(v0::Constant::create(T, Shape{}, {lowest})));
    }

    std::shared_ptr<ov::Node> probs;
    if (sink.get_node()) {
        // Append the per-head sink as one extra logit column, softmax over it, then drop it.
        const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        const auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
        const auto three = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}));
        const auto logits_shape = register_new_node<v3::ShapeOf>(logits);
        const auto sink_shape =
            register_new_node<v0::Concat>(ov::NodeVector{get_dimensions(logits_shape, {0, 1, 2}), one}, 0);
        const auto sink_col = register_new_node<v3::Broadcast>(sink, sink_shape);
        const auto with_sink = register_new_node<v0::Concat>(ov::OutputVector{logits, sink_col}, 3);
        probs = register_new_node<v8::Softmax>(with_sink, -1);
        probs = register_new_node<v8::Slice>(probs, zero, minus_one, one, three);
    } else {
        probs = register_new_node<v8::Softmax>(logits, -1);
    }
    const auto grouped_probs_shape =
        register_new_node<v0::Concat>(ov::NodeVector{batch, kv_heads, minus_one, get_dimensions(q_shape, {2}), kv_len},
                                      0);
    const auto grouped_probs = register_new_node<v1::Reshape>(probs, grouped_probs_shape, false);
    const auto grouped_out = register_new_node<v0::MatMul>(grouped_probs, grouped_v);
    // [B, Nkv, G, S, D] -> [B, N, S, D]
    const auto out_shape = register_new_node<v0::Concat>(ov::NodeVector{batch, num_heads, q_len_head}, 0);
    const auto attention = register_new_node<v1::Reshape>(grouped_out, out_shape, false);
    if (qk_output == QKOutput::AFTER_SOFTMAX) {
        return {attention, probs};
    }
    if (qk_output == QKOutput::BEFORE_SOFTMAX) {
        // ONNX Runtime reports masked (future / out-of-window / beyond-total) logits as 0. Masked entries of the
        // bias-free mask are the type's finite lowest(), so a lowest() / 2 threshold identifies them.
        const auto threshold = register_new_node(v0::Constant::create(T, Shape{}, {finite_lowest(T) / 2}));
        const auto is_masked = register_new_node<v1::Less>(unbiased_mask, threshold);
        const auto typed_zero = register_new_node(v0::Constant::create(T, Shape{}, {0}));
        return {attention, register_new_node<v1::Select>(is_masked, typed_zero, logits)};
    }
    return {attention};
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::windowed_cache_end(
    const ov::Output<ov::Node>& seqlen_scalar,
    const ov::Output<ov::Node>& capacity_scalar,
    int64_t local_window_size) {
    const auto window = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {local_window_size}));
    const auto one_s = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    // gap = capacity - window + 1. The frontend enforces capacity >= local_window_size whenever the capacity
    // is statically known; when it is dynamic that check cannot fire, so clamp gap to >= 1 here to avoid a
    // division by zero (or a negative gap) on a malformed capacity < local_window_size configuration.
    auto gap = register_new_node<v1::Add>(register_new_node<v1::Subtract>(capacity_scalar, window), one_s)->output(0);
    gap = register_new_node<v1::Maximum>(gap, one_s);
    // reclaimed = gap * ceil((x - capacity) / gap), applied only once the cache has overflowed (x > capacity).
    const auto overflow = register_new_node<v1::Subtract>(seqlen_scalar, capacity_scalar);
    const auto ceil_num = register_new_node<v1::Subtract>(register_new_node<v1::Add>(overflow, gap), one_s);
    // Integer division. v1::Divide floors; on the overflowed branch (x > capacity, selected below) ceil_num
    // is always >= 0, so floor and truncation coincide and this yields the intended ceil((x-capacity)/gap).
    const auto blocks = register_new_node<v1::Divide>(ceil_num, gap);
    const auto reclaimed = register_new_node<v1::Multiply>(blocks, gap);
    const auto evicted = register_new_node<v1::Subtract>(seqlen_scalar, reclaimed);
    const auto overflowed = register_new_node<v1::Greater>(seqlen_scalar, capacity_scalar);
    const auto end = register_new_node<v1::Select>(overflowed, evicted, seqlen_scalar);
    // Data-dependent index (feeds Slice/Gather/ScatterUpdate bounds); GPU protects it from fusion.
    end->get_rt_info()["gpu_shape_of_subgraph_root"] = true;
    return end;
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::make_attention_mask(
    const ov::Output<ov::Node>& curr_seqlen_scalar,
    const ov::Output<ov::Node>& kv_len_scalar,
    const ov::Output<ov::Node>& kv_len_1d,
    const ov::Output<ov::Node>& past_seqlen,
    const ov::element::Type& compute_type,
    bool causal,
    int64_t local_window_size,
    const ov::Output<ov::Node>& external_bias,
    const ov::Output<ov::Node>& bias_col_offset,
    [[maybe_unused]] bool sliding_window_cache,
    [[maybe_unused]] float scale,
    [[maybe_unused]] bool has_sink) {
    const bool has_bias = external_bias.get_node_shared_ptr() != nullptr;
    // A window is active for local_window_size >= 1; -1 disables it and 0 is rejected upstream (FE + op).
    // A window is only ever paired with causal=1 (enforced upstream by the FE and the op), so it is only
    // considered on the causal branch below.
    const bool has_window = local_window_size >= 1;

    const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    const auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    const auto two = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));

    // Key positions [1, kv_len]. Coordinates are cache-relative (past_seqlen is the resident past length),
    // which matches the distance-only ONNX Runtime rule.
    const auto zero_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    const auto one_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));
    std::shared_ptr<ov::Node> hori_range =
        register_new_node<v4::Range>(zero_scalar, kv_len_scalar, one_scalar, ov::element::i64);
    hori_range = register_new_node<v0::Unsqueeze>(hori_range, zero);

    std::shared_ptr<ov::Node> masked;
    if (causal) {
        // Absolute query positions [curr, 1]. Causal mask (future keys: k > q), OR-ed with the optional
        // sliding-window band (keys older than the window: (q - k) >= local_window_size). This is applied
        // unconditionally; an external attention_bias is added on top of it, matching ONNX Runtime (the bias
        // does not replace the causal/window mask).
        std::shared_ptr<ov::Node> vert_range =
            register_new_node<v4::Range>(zero_scalar, curr_seqlen_scalar, one_scalar, ov::element::i64);
        vert_range = register_new_node<v0::Unsqueeze>(vert_range, one);
        vert_range = register_new_node<v1::Add>(vert_range, past_seqlen);

        masked = register_new_node<v1::Greater>(hori_range, vert_range);
        if (has_window) {
            const auto distance = register_new_node<v1::Subtract>(vert_range, hori_range);
            const auto window =
                register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {local_window_size}));
            const auto too_old = register_new_node<v1::GreaterEqual>(distance, window);
            masked = register_new_node<v1::LogicalOr>(masked, too_old);
        }
    } else {
        // Bidirectional attention: every query attends to all valid keys. Only the unused cache tail beyond
        // total_sequence_length (past + current) is masked, matching ONNX Runtime's visible_length ==
        // total_seqlen for causal=0. The mask does not depend on the query row, so it broadcasts as [1, kv_len]
        // instead of materializing a full [curr, kv_len] tensor.
        const auto past_scalar = register_new_node<v0::Squeeze>(past_seqlen);
        const auto total_scalar = register_new_node<v1::Add>(past_scalar, curr_seqlen_scalar);
        masked = register_new_node<v1::GreaterEqual>(hori_range, total_scalar);
    }

    const auto typed_zero = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {0}));
    // Finite lowest(), not -inf: a fully-masked row would otherwise softmax to 0/0 = NaN. The magnitude must
    // match the compute type so it does not overflow to -inf when narrowed: f16 and bf16 have far smaller
    // ranges than f32.
    std::shared_ptr<ov::Node> minus_inf;
    if (compute_type == ov::element::f16)
        minus_inf = register_new_node(
            v0::Constant::create(compute_type, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()}));
    else if (compute_type == ov::element::bf16)
        minus_inf = register_new_node(
            v0::Constant::create(compute_type, ov::Shape{}, {std::numeric_limits<ov::bfloat16>::lowest()}));
    else
        minus_inf =
            register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {std::numeric_limits<float>::lowest()}));

    std::shared_ptr<ov::Node> mask = register_new_node<v1::Select>(masked, minus_inf, typed_zero);

    if (has_bias) {
        // Add the external attention_bias [1, num_heads, curr, max_kv] -> [num_heads, curr, kv_len] on top
        // of the causal/window mask (broadcasts over the head axis against the [curr, kv_len] mask). The bias
        // is indexed by absolute key position, so the key window starts at bias_col_offset (0 for a
        // full-length cache; P - resident_rows for a windowed cache, whose first slot is not absolute key 0).
        mask = register_new_node<v1::Add>(
            mask,
            align_attention_bias(external_bias, bias_col_offset, kv_len_1d, compute_type, false));
    }

    return mask;
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::make_batched_attention_mask(
    const ov::Output<ov::Node>& curr_seqlen_scalar,
    const ov::Output<ov::Node>& kv_len_scalar,
    const ov::Output<ov::Node>& kv_len_1d,
    const ov::Output<ov::Node>& past,
    const ov::Output<ov::Node>& total,
    const ov::element::Type& compute_type,
    bool causal,
    int64_t local_window_size,
    const ov::Output<ov::Node>& external_bias) {
    const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    const auto zero_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    const auto one_scalar = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {1}));

    // Everything stays rank 4 ([B, 1, S or 1, kv_len]) so plugins broadcast it in their native layout.
    const auto as_4d = [&](const ov::Output<ov::Node>& value, std::initializer_list<int64_t> shape) {
        return register_new_node<v1::Reshape>(
            value,
            register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, shape)),
            false);
    };
    const auto keys =
        as_4d(register_new_node<v4::Range>(zero_scalar, kv_len_scalar, one_scalar, ov::element::i64), {1, 1, 1, -1});
    std::shared_ptr<ov::Node> masked = register_new_node<v1::GreaterEqual>(keys, as_4d(total, {-1, 1, 1, 1}));
    if (causal) {
        // Absolute query positions [B, 1, S, 1] = past_b + s.
        const auto queries = register_new_node<v1::Add>(
            as_4d(register_new_node<v4::Range>(zero_scalar, curr_seqlen_scalar, one_scalar, ov::element::i64),
                  {1, 1, -1, 1}),
            as_4d(past, {-1, 1, 1, 1}));
        masked = register_new_node<v1::LogicalOr>(masked, register_new_node<v1::Greater>(keys, queries));
        if (local_window_size >= 1) {
            // Window relative to the query position, as in ORT's flash kernel (MLAS flashattn_gqa.cpp).
            const auto window =
                register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {local_window_size}));
            const auto too_old =
                register_new_node<v1::GreaterEqual>(register_new_node<v1::Subtract>(queries, keys), window);
            masked = register_new_node<v1::LogicalOr>(masked, too_old);
        }
    }
    const auto typed_zero = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {0}));
    const auto lowest =
        register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {finite_lowest(compute_type)}));
    std::shared_ptr<ov::Node> mask = register_new_node<v1::Select>(masked, lowest, typed_zero);
    if (external_bias.get_node_shared_ptr()) {
        mask =
            register_new_node<v1::Add>(mask, align_attention_bias(external_bias, zero, kv_len_1d, compute_type, true));
    }
    return mask;
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::align_attention_bias(
    const ov::Output<ov::Node>& bias,
    const ov::Output<ov::Node>& col_offset,
    const ov::Output<ov::Node>& kv_len_1d,
    const ov::element::Type& compute_type,
    bool batched) {
    const auto zero = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    const auto one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    const auto typed_zero = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {0}));
    if (batched) {
        // [B or 1, N or 1, S, total_sequence_length]: take kv_len columns, zero-padding past the bias width (those
        // keys are beyond every batch's total and already masked).
        const auto three = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {3}));
        const auto sliced = register_new_node<v8::Slice>(bias, zero, kv_len_1d, one, three);
        const auto pad_amount = register_new_node<v1::Subtract>(kv_len_1d, get_dimensions(sliced, {3}));
        const auto pads_begin = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, 0}));
        const auto pads_end = register_new_node<v0::Concat>(ov::OutputVector{zero, zero, zero, pad_amount}, 0);
        return register_new_node<v1::Pad>(sliced, pads_begin, pads_end, typed_zero, ov::op::PadMode::CONSTANT);
    }
    // [1, num_heads, curr, max_kv] -> [num_heads, curr, kv_len], broadcasting over the head axis against the
    // [curr, kv_len] mask. The bias is indexed by absolute key position, so the key window starts at col_offset
    // (0 for a full-length cache; P - resident_rows for a windowed cache, whose first slot is not absolute key 0).
    const auto two = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
    const auto squeeze_axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    const auto squeezed = register_new_node<v0::Squeeze>(bias, squeeze_axis);
    const auto bias_stop = register_new_node<v1::Add>(col_offset, kv_len_1d);
    const auto sliced = register_new_node<v8::Slice>(squeezed, col_offset, bias_stop, one, two);
    // The bias only spans total_sequence_length columns, narrower than kv_len whenever K keeps unused trailing rows
    // (a windowed cache below capacity, or a static full-length cache whose buffer exceeds the current total
    // length) - the Slice above then clamps to that narrower width. Zero-pad back up to kv_len: the trailing gap is
    // always past the causal/window edge, so its bias value is never read.
    const auto pad_amount = register_new_node<v1::Subtract>(kv_len_1d, get_dimensions(sliced, {2}));
    const auto pads_begin = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, 0}));
    const auto pads_end = register_new_node<v0::Concat>(ov::OutputVector{zero, zero, pad_amount}, 0);
    return register_new_node<v1::Pad>(sliced, pads_begin, pads_end, typed_zero, ov::op::PadMode::CONSTANT);
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::get_dimensions(
    const std::shared_ptr<v3::ShapeOf>& shape,
    const std::vector<int>& dims) {
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return register_new_node<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::get_dimensions(
    const std::shared_ptr<ov::Node>& node,
    const std::vector<int>& dims) {
    return get_dimensions(register_new_node<v3::ShapeOf>(node), dims);
}

ov::Output<ov::Node> ov::pass::GroupQueryAttentionDecomposition::make_qk_norm(const ov::Output<ov::Node>& input,
                                                                              const ov::Output<ov::Node>& weight,
                                                                              float epsilon) {
    const auto type = input.get_element_type();
    const bool upcast = type != ov::element::f32;
    const auto x = upcast ? register_new_node<v0::Convert>(input, ov::element::f32)->output(0) : input;
    const auto gamma = upcast ? register_new_node<v0::Convert>(weight, ov::element::f32)->output(0) : weight;
    const auto two = register_new_node(v0::Constant::create(ov::element::f32, ov::Shape{}, {2.0f}));
    const auto axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    const auto eps = register_new_node(v0::Constant::create(ov::element::f32, ov::Shape{}, {epsilon}));
    const auto mean = register_new_node<v1::ReduceMean>(register_new_node<v1::Power>(x, two), axis, true);
    const auto rms = register_new_node<v0::Sqrt>(register_new_node<v1::Add>(mean, eps));
    const auto normed = register_new_node<v1::Multiply>(gamma, register_new_node<v1::Divide>(x, rms));
    return upcast ? register_new_node<v0::Convert>(normed, type)->output(0) : normed->output(0);
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::rotaryEmbedding(ov::Output<ov::Node> input,
                                                                                      ov::Output<ov::Node> cos,
                                                                                      ov::Output<ov::Node> sin,
                                                                                      bool interleaved) {
    auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    // rotary_dim (2 * cos.shape[-1]) may be smaller than head_size for GPT-NeoX/Phi-style partial RoPE:
    // only the leading rotary_dim channels are rotated below; the trailing channels pass through
    // unchanged. The op-level validate_and_infer_types() already bounds rotary_dim <= head_size.
    const auto& cos_partial_shape = cos.get_partial_shape();
    const auto half_head_size_val =
        static_cast<int64_t>(cos_partial_shape[cos_partial_shape.rank().get_length() - 1].get_length());
    const auto rotary_dim_val = 2 * half_head_size_val;
    const auto& input_partial_shape = input.get_partial_shape();
    const auto head_size_val =
        static_cast<int64_t>(input_partial_shape[input_partial_shape.rank().get_length() - 1].get_length());
    const bool is_partial_rotary = rotary_dim_val < head_size_val;

    ov::Output<ov::Node> rotary_input = input;
    if (is_partial_rotary) {
        // Slice out only the leading rotary_dim channels to feed the RoPE math below; the trailing
        // pass-through channels are never materialized as a separate tensor - re-attaching them later is a
        // ScatterUpdate into the original `input`, not a Concat (avoids holding a live pass_through copy).
        const auto slice_start = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        const auto slice_stop = v0::Constant::create(ov::element::i64, ov::Shape{1}, {rotary_dim_val});
        const auto slice_step = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto slice_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        rotary_input = register_new_node<v8::Slice>(input, slice_start, slice_stop, slice_step, slice_axis);
    }

    // Unsqueeze cos/sin to 4D [1, 1, seqlen, head_size/2] to match RoPE fusion pattern
    // Batched [B, S, D/2] cos/sin only gain the head axis: [B, 1, S, D/2].
    auto unsqueeze_axes = cos_partial_shape.rank().get_length() == 3
                              ? v0::Constant::create(ov::element::i64, ov::Shape{1}, {1})
                              : v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto cos_4d = register_new_node<v0::Unsqueeze>(cos, unsqueeze_axes);
    auto sin_4d = register_new_node<v0::Unsqueeze>(sin, unsqueeze_axes);

    // For interleaved mode, deinterleave first so the core RoPE formula is identical
    ov::Output<ov::Node> rope_input = rotary_input;
    std::shared_ptr<v3::ShapeOf> input_shape;
    std::shared_ptr<ov::Node> dim_bns, half_head_size;
    std::shared_ptr<v0::Constant> perm_5d;
    if (interleaved) {
        input_shape = register_new_node<v3::ShapeOf>(rotary_input);
        dim_bns = get_dimensions(input_shape, {0, 1, 2});
        half_head_size = get_dimensions(cos.get_node_shared_ptr(), {-1});
        perm_5d = v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});

        // Deinterleave: [bs,nh,seq,rotary_dim]
        //   -> reshape [bs,nh,seq,rotary_dim/2,2]
        //   -> transpose [bs,nh,seq,2,rotary_dim/2]
        //   -> reshape [bs,nh,seq,rotary_dim]  (now [first_half, second_half])
        auto deinterleave_5d = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, half_head_size, two}, 0);
        auto reshaped_5d = register_new_node<v1::Reshape>(rotary_input, deinterleave_5d, false);
        auto transposed_5d = register_new_node<v1::Transpose>(reshaped_5d, perm_5d);
        rope_input = register_new_node<v1::Reshape>(transposed_5d, input_shape, false);
    }

    // Core RoPE formula (matches RoPEFusionGPTOSS pattern for both modes)
    // first_ = first_half * cos - second_half * sin
    // second_ = second_half * cos + first_half * sin
    const auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    const auto split_lengths =
        v0::Constant::create(ov::element::i64, ov::Shape{2}, {half_head_size_val, half_head_size_val});
    // Split along last axis using constant split_lengths to enable RoPE fusion pattern matching
    auto in_split = register_new_node<v1::VariadicSplit>(rope_input, split_axis, split_lengths)->outputs();
    auto first_half_mul_cos = register_new_node<v1::Multiply>(in_split[0], cos_4d);
    auto second_half_mul_sin = register_new_node<v1::Multiply>(in_split[1], sin_4d);
    auto neg_one = v0::Constant::create(input.get_element_type(), ov::Shape{}, {-1.0f});
    auto neg_second_sin = register_new_node<v1::Multiply>(second_half_mul_sin, neg_one);
    auto res_0 = register_new_node<v1::Add>(first_half_mul_cos, neg_second_sin);
    auto second_half_mul_cos = register_new_node<v1::Multiply>(in_split[1], cos_4d);
    auto first_half_mul_sin = register_new_node<v1::Multiply>(in_split[0], sin_4d);
    auto res_1 = register_new_node<v1::Add>(second_half_mul_cos, first_half_mul_sin);
    ov::Output<ov::Node> output = register_new_node<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);

    // For interleaved mode, re-interleave the result
    if (interleaved) {
        // Re-interleave: [bs,nh,seq,rotary_dim]
        //   -> reshape [bs,nh,seq,2,rotary_dim/2]
        //   -> transpose [bs,nh,seq,rotary_dim/2,2]
        //   -> reshape [bs,nh,seq,rotary_dim]
        auto reinterleave_5d = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, two, half_head_size}, 0);
        auto result_5d = register_new_node<v1::Reshape>(output, reinterleave_5d, false);
        auto result_transposed = register_new_node<v1::Transpose>(result_5d, perm_5d);
        output = register_new_node<v1::Reshape>(result_transposed, input_shape, false);
    }

    if (is_partial_rotary) {
        // Scatter the rotated channels back into `input` at [0, rotary_dim); channels beyond rotary_dim
        // are left untouched since they were never sliced out.
        const auto zero_s = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        const auto one_s = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        const auto rotary_dim_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, {rotary_dim_val});
        const auto scatter_indices = register_new_node<v4::Range>(zero_s, rotary_dim_scalar, one_s, ov::element::i64);
        const auto scatter_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        output = register_new_node<v3::ScatterUpdate>(input, scatter_indices, output, scatter_axis);
    }

    return output.get_node_shared_ptr();
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::make_kv_scale(
    const ov::Output<ov::Node>& scale,
    int64_t kv_num_heads,
    ov::op::internal::GroupQueryAttentionQuantType quant_type) {
    // The KV cache is laid out as [batch, kv_num_heads, seq_len, head_size]. Reshape the flat scale so it
    // broadcasts along that layout. A fully static target shape is used (no -1 wildcard) so the result stays
    // static-shaped for plugins (e.g. NPU) that require static shapes. A fresh shape Constant is built per
    // call so no two GQA layers alias it.
    std::vector<int64_t> target_shape;
    if (quant_type == ov::op::internal::GroupQueryAttentionQuantType::PER_CHANNEL) {
        // Per-channel scale has kv_num_heads * head_size elements, head-major (scale[kv_head * head_size + ch]).
        // Reshape to [1, kv_num_heads, 1, head_size] to broadcast over batch and seq. head_size is derived from
        // the (static) scale length when known, otherwise falls back to a -1 wildcard.
        int64_t head_size = -1;
        const auto& scale_pshape = scale.get_partial_shape();
        if (scale_pshape.is_static()) {
            // Element count alone determines head_size, regardless of rank/dim order (e.g. rank 1
            // [kv_num_heads*head_size], rank 2 [kv_num_heads,head_size], rank 4
            // [1,kv_num_heads,1,head_size] all resolve the same way).
            head_size = static_cast<int64_t>(ov::shape_size(scale_pshape.to_shape())) / kv_num_heads;
        }
        target_shape = {1, kv_num_heads, 1, head_size};
    } else {
        // PER_TENSOR: a single scalar broadcast over the whole tensor.
        target_shape = {1, 1, 1, 1};
    }
    const auto shape_const =
        register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{target_shape.size()}, target_shape));
    return register_new_node<v1::Reshape>(scale, shape_const, false);
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::dequantize_kv(
    const ov::Output<ov::Node>& quantized,
    const ov::Output<ov::Node>& scale,
    int64_t kv_num_heads,
    int64_t kv_cache_bit_width,
    ov::op::internal::GroupQueryAttentionQuantType quant_type,
    const ov::element::Type& compute_type) {
    // Symmetric dequantization matching ONNX Runtime MLAS/CUDA QDQ. The actual Convert(->float) * scale
    // (optionally - zero_point) chain is built via the shared ov::decomposition::low_precision_dequantize
    // helper, so it produces the canonical dequantization pattern recognized by MarkDequantization / LPT.
    const auto scale_bcast = make_kv_scale(scale, kv_num_heads, quant_type);

    if (kv_cache_bit_width == 4) {
        // 4-bit cache is stored as u8 with two signed values packed per byte (even channel -> low nibble,
        // odd channel -> high nibble) biased by +8. Unpack the nibbles back into per-channel integers first,
        // then let low_precision_dequantize apply the (Convert - 8) * scale chain (zero_point = 8 removes the bias).
        const auto axis_last = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        const auto mask_low =
            register_new_node(v0::Constant::create(quantized.get_element_type(), ov::Shape{}, {0x0F}));
        const auto shift_4 = register_new_node(v0::Constant::create(quantized.get_element_type(), ov::Shape{}, {4}));
        const auto low_nibble = register_new_node<v13::BitwiseAnd>(quantized, mask_low);
        const auto high_nibble = register_new_node<v15::BitwiseRightShift>(quantized, shift_4);
        // Interleave low/high nibbles back along the head_size axis: [.., packed] -> [.., packed, 2] -> [.., 2*packed].
        const auto low_u = register_new_node<v0::Unsqueeze>(low_nibble, axis_last);
        const auto high_u = register_new_node<v0::Unsqueeze>(high_nibble, axis_last);
        const auto interleaved = register_new_node<v0::Concat>(ov::OutputVector{low_u, high_u}, -1);
        const auto flat_shape = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, -1}));
        const auto unpacked = register_new_node<v1::Reshape>(interleaved, flat_shape, true);
        const auto zero_point = register_new_node(v0::Constant::create(quantized.get_element_type(), ov::Shape{}, {8}));
        ov::pass::NodeRegistry reg;
        auto dequant =
            ov::decomposition::low_precision_dequantize(reg, unpacked, scale_bcast, zero_point, {}, compute_type);
        for (const auto& node : reg.get()) {
            register_new_node(node);
        }
        return dequant.get_node_shared_ptr();
    }

    // 8-bit cache: symmetric (no zero point) Convert(->compute) * scale via the shared helper.
    ov::pass::NodeRegistry reg;
    auto dequant = ov::decomposition::low_precision_dequantize(reg, quantized, scale_bcast, {}, {}, compute_type);
    for (const auto& node : reg.get()) {
        register_new_node(node);
    }
    return dequant.get_node_shared_ptr();
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::quantize_kv(
    const ov::Output<ov::Node>& current,
    const ov::Output<ov::Node>& scale,
    int64_t kv_num_heads,
    int64_t kv_cache_bit_width,
    ov::op::internal::GroupQueryAttentionQuantType quant_type,
    const ov::element::Type& cache_type) {
    // Symmetric quantize-on-write: q = clamp(round(x * inv_scale)). Rounding is round-half-to-even to match the
    // ONNX Runtime MLAS/CUDA reference (std::rintf). Clamp is applied before the narrowing Convert to avoid
    // overflow on out-of-range values. inv_scale mirrors MLAS SafeInvScale: a zero scale maps to 1.0 so the step
    // degenerates to clamp(round(x)) instead of producing NaN (and multiplying by the reciprocal matches MLAS).
    const auto compute_type = current.get_element_type();
    const auto scale_bcast = make_kv_scale(scale, kv_num_heads, quant_type);
    ov::Output<ov::Node> scale_ct = scale_bcast;
    if (scale.get_element_type() != compute_type) {
        scale_ct = register_new_node<v0::Convert>(scale_bcast, compute_type);
    }
    const auto one = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {1}));
    const auto zero = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {0}));
    const auto is_zero_scale = register_new_node<v1::Equal>(scale_ct, zero);
    const auto safe_scale = register_new_node<v1::Select>(is_zero_scale, one, scale_ct);
    const auto inv_scale = register_new_node<v1::Divide>(one, safe_scale);
    const auto scaled = register_new_node<v1::Multiply>(current, inv_scale);

    if (cache_type == ov::element::f8e4m3) {
        // f8e4m3 cache: no integer Round (Convert rounds to the f8e4m3 grid). Clamp to +/-448 (f8e4m3 max)
        // first: Convert to f8e4m3 maps out-of-range magnitudes to NaN, not saturation.
        const auto clamped = register_new_node<v0::Clamp>(scaled, -448.0, 448.0);
        return register_new_node<v0::Convert>(clamped, cache_type);
    }

    const auto rounded = register_new_node<v5::Round>(scaled, v5::Round::RoundMode::HALF_TO_EVEN);

    if (kv_cache_bit_width == 4) {
        // Clamp to signed 4-bit range, add the +8 storage bias, then pack pairs of channels into u8 bytes.
        const auto clamped = register_new_node<v0::Clamp>(rounded, -8.0, 7.0);
        const auto bias = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {8}));
        const auto biased = register_new_node<v1::Add>(clamped, bias);
        const auto as_u8 = register_new_node<v0::Convert>(biased, cache_type);
        // Split the head_size axis into pairs: [.., 2*packed] -> [.., packed, 2] -> low/high nibbles.
        const auto pair_shape =
            register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 0, 0, -1, 2}));
        const auto paired = register_new_node<v1::Reshape>(as_u8, pair_shape, true);
        const auto axis_last = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
        const auto split = register_new_node<v1::VariadicSplit>(
            paired,
            register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{}, {-1})),
            register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 1})));
        const auto low = register_new_node<v0::Squeeze>(split->output(0), axis_last);
        const auto high = register_new_node<v0::Squeeze>(split->output(1), axis_last);
        const auto shift_4 = register_new_node(v0::Constant::create(cache_type, ov::Shape{}, {4}));
        const auto mask_low = register_new_node(v0::Constant::create(cache_type, ov::Shape{}, {0x0F}));
        const auto low_masked = register_new_node<v13::BitwiseAnd>(low, mask_low);
        const auto high_shifted = register_new_node<v15::BitwiseLeftShift>(high, shift_4);
        return register_new_node<v13::BitwiseOr>(low_masked, high_shifted);
    }

    // 8-bit cache: clamp to signed 8-bit range and narrow to the cache type.
    const auto clamped = register_new_node<v0::Clamp>(rounded, -128.0, 127.0);
    return register_new_node<v0::Convert>(clamped, cache_type);
}
