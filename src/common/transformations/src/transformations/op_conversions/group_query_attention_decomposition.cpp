// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <memory>

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
#include "openvino/op/logical_or.hpp"
#include "openvino/op/maximum.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/round.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
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
    // TODO: add softcap support

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
    const auto seqlens_elemi64 = register_new_node<v0::Convert>(seqlens_k, ov::element::i64);
    const auto real_seqlens = register_new_node<v1::Add>(seqlens_elemi64, one);

    // Only consider batch is 1
    const auto seqlens_1d = register_new_node<v1::Reshape>(real_seqlens, one, false);
    const auto past_seqlen = register_new_node<v1::Subtract>(seqlens_1d, current_seqlen);
    const auto curr_seqlen_scalar = register_new_node<v0::Squeeze>(current_seqlen);

    if (do_rotary) {
        // Get cos_cache and sin_cache from their actual input indices (ONNX COS_CACHE and SIN_CACHE).
        // validate_and_infer_types() already verified these inputs exist and indices are valid when do_rotary is true.
        auto cos_cache = get_input(GQAInputs::COS_CACHE);
        auto sin_cache = get_input(GQAInputs::SIN_CACHE);

        ov::Output<ov::Node> position_ids =
            register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
        // Check if position_ids is provided (optional input), using actual input index
        if (has_input(GQAInputs::POSITION_IDS)) {
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
        K = rotaryEmbedding(K, cos, sin, rotary_interleaved);
    }
    const auto is_static_input = K.get_partial_shape().is_static() && past_key.get_partial_shape().is_static();

    // Quantize-on-write: when the cache is quantized, quantize the (post-RoPE) current K/V into the cache type
    // before appending them, so the assembled present cache stays quantized and the past bytes are preserved
    // verbatim (no re-rounding of past tokens). Matches ONNX Runtime MLAS/CUDA semantics.
    if (kv_quantized) {
        K = quantize_kv(K, k_scale, kv_num_heads, kv_cache_bit_width, k_quant_type, kv_cache_type);
        V = quantize_kv(V, v_scale, kv_num_heads, kv_cache_bit_width, v_quant_type, kv_cache_type);
    }

    // past_seqlen expressed in the coordinate system the attention mask uses. Equals the absolute past
    // length for a full-length cache; a windowed cache overrides it with the resident row count.
    ov::Output<ov::Node> mask_past_seqlen = past_seqlen;
    // Absolute key position of the KV buffer's first slot, used to align an external attention_bias (indexed
    // by absolute key). 0 for a full-length cache (slot j == absolute key j); a windowed cache rolls, so its
    // first slot holds absolute key P - resident_rows (set in the windowed branches below).
    ov::Output<ov::Node> bias_col_offset = zero;
    ov::Output<ov::Node> present_k, present_v;

    if (node->get_sliding_window_cache()) {
        // Windowed KV cache (capacity C, rolled with front eviction). end_before/end_after are the resident
        // row counts before/after appending the S new tokens (see windowed_cache_end).
        const auto capacity = get_dimensions(past_key.get_node_shared_ptr(), {2});
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

            K = present_k;
            V = present_v;
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
        past_key = register_new_node<v8::Slice>(past_key, zero, past_seqlen, one, two);
        past_value = register_new_node<v8::Slice>(past_value, zero, past_seqlen, one, two);
        K = construct_kv_cache(past_key, K);
        V = construct_kv_cache(past_value, V);
        present_k = K;
        present_v = V;
    }

    // Dequantize the assembled cache to the compute (float) type for the attention math. Everything downstream
    // (head broadcast, mask, SDPA) then operates in float exactly as in the non-quantized path.
    if (kv_quantized) {
        K = dequantize_kv(K, k_scale, kv_num_heads, kv_cache_bit_width, k_quant_type, T);
        V = dequantize_kv(V, v_scale, kv_num_heads, kv_cache_bit_width, v_quant_type, T);
    }

    const auto concat_kv_len = get_dimensions(K.get_node_shared_ptr(), {2});
    const auto concat_kv_len_scalar = register_new_node<v0::Squeeze>(concat_kv_len);

    // Broadcast KV if grouped query attention
    const size_t kv_num_heads_factor = num_heads / kv_num_heads;
    if (kv_num_heads_factor > 1) {
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
    const auto mask = make_attention_mask(curr_seqlen_scalar,
                                          concat_kv_len_scalar,
                                          concat_kv_len,
                                          mask_past_seqlen,
                                          T,
                                          causal,
                                          local_window_size,
                                          external_bias,
                                          bias_col_offset,
                                          node->get_sliding_window_cache(),
                                          scale,
                                          has_sink);

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

    std::shared_ptr<ov::Node> qga_output;
    if (sink.get_node_shared_ptr()) {
        // SDPA's 6-input form requires an explicit scale; use the op scale or the default 1/sqrt(head_size).
        ov::Output<ov::Node> scale_node;
        if (scale != 0.0f) {
            scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
        } else {
            const auto head_size_t = register_new_node<v0::Convert>(head_size_node, T);
            const auto neg_half = register_new_node(v0::Constant::create(T, Shape{}, {-0.5f}));
            scale_node = register_new_node<v0::Squeeze>(register_new_node<ov::op::v1::Power>(head_size_t, neg_half));
        }
        qga_output = make_sdpa(Q, K, V, mask, scale_node, sink, false);
    } else if (scale != 0.0f) {
        auto scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
        qga_output = make_sdpa(Q, K, V, mask, scale_node, {}, false);
    } else {
        qga_output = make_sdpa(Q, K, V, mask, {}, {}, !mask);
    }

    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads * head_size)
    auto perm = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
    auto qga_output_transposed = register_new_node<v1::Transpose>(qga_output, perm);
    auto dim_merge_shape = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1}));
    auto output = register_new_node<v1::Reshape>(qga_output_transposed, dim_merge_shape, true)->output(0);

    return {output, present_k, present_v};
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::make_sdpa(const ov::Output<ov::Node>& query,
                                                                                const ov::Output<ov::Node>& key,
                                                                                const ov::Output<ov::Node>& value,
                                                                                const ov::Output<ov::Node>& mask,
                                                                                const ov::Output<ov::Node>& scale,
                                                                                const ov::Output<ov::Node>& sink,
                                                                                bool is_causal) {
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
    // ranges than f32. (The core op currently restricts the activation type to {f32, f16}, so bf16 is not
    // yet reachable here, but keep an explicit branch so a future bf16 activation stays finite.)
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
        const auto squeeze_axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        std::shared_ptr<ov::Node> bias = register_new_node<v0::Squeeze>(external_bias, squeeze_axis);
        const auto bias_stop = register_new_node<v1::Add>(bias_col_offset, kv_len_1d);
        bias = register_new_node<v8::Slice>(bias, bias_col_offset, bias_stop, one, two);

        // The bias only spans total_sequence_length columns, narrower than kv_len whenever K keeps unused
        // trailing rows (a windowed cache below capacity, or a static full-length cache whose buffer
        // exceeds the current total length) - the Slice above then clamps to that narrower width. Zero-pad
        // back up to kv_len: the trailing gap is always past the causal/window edge (every branch above
        // places resident/new rows before it), so its bias value is never read.
        const auto bias_kv_len = get_dimensions(bias, {2});
        const auto pad_amount = register_new_node<v1::Subtract>(kv_len_1d, bias_kv_len);
        const auto pads_begin = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, 0}));
        const auto pads_end = register_new_node<v0::Concat>(ov::OutputVector{zero, zero, pad_amount}, 0);
        bias = register_new_node<v1::Pad>(bias, pads_begin, pads_end, typed_zero, ov::op::PadMode::CONSTANT);

        mask = register_new_node<v1::Add>(mask, bias);
    }

    return mask;
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
    auto unsqueeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
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
