// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/paged_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/logical_or.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/mod.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_nd_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
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
namespace v8 = ov::op::v8;
namespace v13 = ov::op::v13;

ov::pass::PagedAttentionDecomposition::PagedAttentionDecomposition() {
    MATCHER_SCOPE(PagedAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::internal::PagedAttention>();

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        auto node = ov::as_type_ptr<ov::op::internal::PagedAttention>(m.get_match_root());
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

ov::OutputVector ov::pass::PagedAttentionDecomposition::decompose(
    std::shared_ptr<ov::op::internal::PagedAttention> node) {
    const auto num_heads = node->get_num_heads();
    const auto kv_num_heads = node->get_kv_num_heads();
    const auto scale = node->get_scale();
    const auto softcap = node->get_softcap();
    const auto local_window_size = node->get_local_window_size();
    const auto do_rotary = node->get_do_rotary();
    const auto rotary_interleaved = node->get_rotary_interleaved();

    // Inputs (attribute-determined arity, set by the frontend): Q, K, V (all 2-D [num_tokens, heads*head_size]),
    // key_cache, value_cache ([num_blocks, block_size, kv_num_heads, head_size]), cumulative_sequence_length
    // ([batch+1] i32), past_seqlens ([batch] i32), block_table ([batch, max_blocks] i32), and cos/sin caches
    // when do_rotary. This decomposition handles the single-sequence (batch == 1) case.
    auto Q = node->input_value(0);
    auto K = node->input_value(1);
    auto V = node->input_value(2);
    auto key_cache = node->input_value(3);
    auto value_cache = node->input_value(4);
    auto cumulative_sequence_length = node->input_value(5);
    auto past_seqlens = node->input_value(6);
    auto block_table = node->input_value(7);

    const auto element_type = Q.get_element_type();

    // --- i64 constants (shape-building world) ---
    const auto zero_i64 = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    const auto one_i64 = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    const auto neg_one_i64 = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
    // --- i32 constants (paged-index world; all PA metadata is i32) ---
    const auto zero_i32_s = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
    const auto one_i32_s = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {1}));
    const auto axis0_i32 = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));

    // === Step 0: per-sequence scalars (batch == 1), all i32 ===
    // T = number of new query tokens = cumulative_sequence_length[1] - cumulative_sequence_length[0].
    const auto cum1 =
        register_new_node<v8::Gather>(cumulative_sequence_length,
                                      register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {1})),
                                      axis0_i32);
    const auto cum0 = register_new_node<v8::Gather>(cumulative_sequence_length, zero_i32_s, axis0_i32);
    const auto T_scalar = register_new_node<v1::Subtract>(cum1, cum0);  // i32 scalar
    // past_len = past_seqlens[0]; total_ctx = past_len + T.
    const auto past_len_scalar = register_new_node<v8::Gather>(past_seqlens, zero_i32_s, axis0_i32);  // i32 scalar
    const auto total_ctx_scalar = register_new_node<v1::Add>(past_len_scalar, T_scalar);              // i32 scalar
    // block_table row 0: [max_blocks] i32.
    const auto bt_row0 = register_new_node<v8::Gather>(block_table, zero_i32_s, axis0_i32);
    // block_size from key_cache dim 1 ([num_blocks, block_size, kv_num_heads, head_size]).
    const auto kc_shape = register_new_node<v3::ShapeOf>(key_cache);  // i64
    const auto block_size_i64 = register_new_node<v0::Squeeze>(get_dimensions(kc_shape, {1}));
    const auto block_size_i32_s = register_new_node<v0::Convert>(block_size_i64, ov::element::i32);

    // === Step 1: unpack 2-D [num_tokens, heads*head_size] Q/K/V into SDPA layout [1, heads, T, head_size] ===
    // head_size (i64) from Q's hidden dim (dim 1) divided by num_heads.
    const auto q_shape = register_new_node<v3::ShapeOf>(Q);
    const auto q_hidden = get_dimensions(q_shape, {1});  // [num_heads * head_size]
    const auto head_size_i64 = register_new_node<v1::Divide>(
        q_hidden,
        register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads})));
    const auto perm = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
    const auto T_i64 = register_new_node<v0::Convert>(register_new_node<v0::Unsqueeze>(T_scalar, zero_i32_s),
                                                      ov::element::i64);  // [1]

    auto to_sdpa_layout = [&](const ov::Output<ov::Node>& x, int64_t heads) {
        // [num_tokens, heads*head_size] -> [1, num_tokens, heads, head_size] -> transpose -> [1, heads, T, head_size].
        const auto shape = register_new_node<v0::Concat>(
            ov::NodeVector{one_i64,
                           T_i64,
                           register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {heads})),
                           head_size_i64},
            0);
        const auto reshaped = register_new_node<v1::Reshape>(x, shape, false);
        return register_new_node<v1::Transpose>(reshaped, perm)->output(0);
    };
    Q = to_sdpa_layout(Q, num_heads);
    K = to_sdpa_layout(K, kv_num_heads);
    V = to_sdpa_layout(V, kv_num_heads);

    // === Step 2: RoPE on Q and the new K (positions past_len + [0, T)), before the K/V are written to cache ===
    if (do_rotary) {
        auto cos_cache = node->input_value(node->get_input_size() - 2);
        auto sin_cache = node->input_value(node->get_input_size() - 1);
        // position_ids = past_len + Range(0, T) : [T] i32.
        const auto range_t = register_new_node<v4::Range>(zero_i32_s, T_scalar, one_i32_s, ov::element::i32);
        const auto position_ids = register_new_node<v1::Add>(range_t, past_len_scalar);
        const auto cos = register_new_node<v8::Gather>(cos_cache, position_ids, zero_i32_s);
        const auto sin = register_new_node<v8::Gather>(sin_cache, position_ids, zero_i32_s);
        Q = rotaryEmbedding(Q, cos, sin, rotary_interleaved);
        K = rotaryEmbedding(K, cos, sin, rotary_interleaved);
    }

    // === Step 3: write the new K/V into the paged cache -> key_cache_out / value_cache_out ===
    // Flatten cache [num_blocks, block_size, kv_num_heads, head_size] -> [num_blocks*block_size, kv_num_heads,
    // head_size] and ScatterNDUpdate the new tokens at their physical slots (non-contiguous -> ScatterNDUpdate,
    // not ScatterUpdate). New K/V go to layout [T, kv_num_heads, head_size] to match the flattened cache rows.
    const auto flat_pattern =
        register_new_node<v0::Concat>(ov::NodeVector{neg_one_i64, get_dimensions(kc_shape, {2, 3})}, 0);
    const auto kc_flat = register_new_node<v1::Reshape>(key_cache, flat_pattern, false);
    const auto vc_flat = register_new_node<v1::Reshape>(value_cache, flat_pattern, false);

    // Physical slot indices [T, 1] for the new tokens (logical positions [past_len, total_ctx)).
    const auto write_slots = build_slot_indices(bt_row0, past_len_scalar, T_scalar, block_size_i32_s);
    const auto write_indices = register_new_node<v0::Unsqueeze>(write_slots, neg_one_i64);  // [T, 1]

    auto to_cache_rows = [&](const ov::Output<ov::Node>& x) {
        // [1, kv_num_heads, T, head_size] -> transpose [1, T, kv_num_heads, head_size] -> squeeze -> [T, kv, head].
        const auto t = register_new_node<v1::Transpose>(x, perm);
        return register_new_node<v0::Squeeze>(t, zero_i64)->output(0);
    };
    const auto k_rows = to_cache_rows(K);
    const auto v_rows = to_cache_rows(V);
    const auto kc_flat_new = register_new_node<v3::ScatterNDUpdate>(kc_flat, write_indices, k_rows);
    const auto vc_flat_new = register_new_node<v3::ScatterNDUpdate>(vc_flat, write_indices, v_rows);
    // Restore the paged cache shape for the outputs (key/value caches share the same shape).
    const auto key_cache_out = register_new_node<v1::Reshape>(kc_flat_new, kc_shape, false);
    const auto value_cache_out =
        register_new_node<v1::Reshape>(vc_flat_new, register_new_node<v3::ShapeOf>(value_cache), false);

    // === Step 4: gather the full KV context [total_ctx, kv, head] from the (updated) cache ===
    const auto read_slots = build_slot_indices(bt_row0, zero_i32_s, total_ctx_scalar, block_size_i32_s);
    const auto k_ctx_rows =
        register_new_node<v8::Gather>(kc_flat_new, read_slots, zero_i32_s);  // [total_ctx, kv, head]
    const auto v_ctx_rows = register_new_node<v8::Gather>(vc_flat_new, read_slots, zero_i32_s);
    auto ctx_to_sdpa = [&](const ov::Output<ov::Node>& rows) {
        // [total_ctx, kv, head] -> [kv, total_ctx, head] -> [1, kv, total_ctx, head].
        const auto p = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, 0, 2}));
        const auto t = register_new_node<v1::Transpose>(rows, p);
        return register_new_node<v0::Unsqueeze>(t, zero_i64)->output(0);
    };
    K = ctx_to_sdpa(k_ctx_rows);
    V = ctx_to_sdpa(v_ctx_rows);

    // === Step 5: broadcast KV heads for grouped-query attention (num_heads / kv_num_heads > 1) ===
    const size_t kv_num_heads_factor = num_heads / kv_num_heads;
    if (kv_num_heads_factor > 1) {
        const auto kv_shape = register_new_node<v3::ShapeOf>(K);
        const auto kv_prev_2 = get_dimensions(kv_shape, {0, 1});
        const auto kv_last_2 = get_dimensions(kv_shape, {2, 3});
        const auto expand_shape = register_new_node<v0::Concat>(ov::NodeVector{kv_prev_2, one_i64, kv_last_2}, 0);
        K = register_new_node<v1::Reshape>(K, expand_shape, false);
        V = register_new_node<v1::Reshape>(V, expand_shape, false);
        K = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, K), 2);
        V = register_new_node<v0::Concat>(ov::OutputVector(kv_num_heads_factor, V), 2);
        const auto q_prev_2 = get_dimensions(register_new_node<v3::ShapeOf>(Q), {0, 1});
        const auto final_shape = register_new_node<v0::Concat>(ov::NodeVector{q_prev_2, kv_last_2}, 0);
        K = register_new_node<v1::Reshape>(K, final_shape, false);
        V = register_new_node<v1::Reshape>(V, final_shape, false);
    }

    // === Step 6: causal + sliding-window additive mask [T, total_ctx] ===
    const auto mask = make_attention_mask(T_scalar, total_ctx_scalar, past_len_scalar, element_type, local_window_size);

    // === Step 7: attention core (mask encodes causal + past offset + window) + repack ===
    // Without softcap, use ScaledDotProductAttention (causal=false: the mask already carries causal + past
    // offset + window). With softcap, ScaledDotProductAttention has no soft-capping, so build the attention
    // manually: scale -> softcap(softcap * tanh(scores / softcap)) -> mask -> softmax -> @ V, matching the
    // ONNX Runtime PagedAttention reference order.
    std::shared_ptr<ov::Node> sdpa_out;
    if (softcap > 0.0f) {
        sdpa_out = build_attention_softcap(Q, K, V, mask, scale, softcap, element_type);
    } else if (scale != 0.0f) {
        const auto scale_node = register_new_node(v0::Constant::create(element_type, ov::Shape{}, {scale}));
        sdpa_out = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, scale_node, false);
    } else {
        sdpa_out = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, false);
    }
    // [1, num_heads, T, head_size] -> transpose [1, T, num_heads, head_size] -> reshape [1, T, num_heads*head_size]
    // -> squeeze batch -> [num_tokens, num_heads * head_size].
    const auto out_t = register_new_node<v1::Transpose>(sdpa_out, perm);
    const auto merge_shape = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1}));
    const auto out_3d = register_new_node<v1::Reshape>(out_t, merge_shape, true);
    const auto output = register_new_node<v0::Squeeze>(out_3d, zero_i64);

    return {output, key_cache_out, value_cache_out};
}

std::shared_ptr<ov::Node> ov::pass::PagedAttentionDecomposition::build_attention_softcap(
    const ov::Output<ov::Node>& Q,
    const ov::Output<ov::Node>& K,
    const ov::Output<ov::Node>& V,
    const ov::Output<ov::Node>& mask,
    float scale,
    float softcap,
    const ov::element::Type& compute_type) {
    // Manual attention core for the softcap path (ScaledDotProductAttention has no soft-capping). Q/K/V are
    // [1, num_heads, T/total_ctx, head_size]. Order matches ONNX Runtime attention_ref: scale -> softcap ->
    // mask -> softmax -> @ V. Returns [1, num_heads, T, head_size].
    // scores = Q @ K^T
    auto scores = register_new_node<ov::op::v0::MatMul>(Q, K, false, true)->output(0);
    // scale: explicit op scale, else 1/sqrt(head_size).
    if (scale != 0.0f) {
        const auto scale_node = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {scale}));
        scores = register_new_node<v1::Multiply>(scores, scale_node);
    } else {
        const auto head_size =
            register_new_node<v0::Convert>(get_dimensions(Q.get_node_shared_ptr(), {3}), compute_type);
        const auto neg_half = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {-0.5f}));
        const auto inv_sqrt = register_new_node<v0::Squeeze>(register_new_node<ov::op::v1::Power>(head_size, neg_half));
        scores = register_new_node<v1::Multiply>(scores, inv_sqrt);
    }
    // softcap: softcap * tanh(scores / softcap).
    const auto cap = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {softcap}));
    const auto capped =
        register_new_node<v1::Multiply>(register_new_node<ov::op::v0::Tanh>(register_new_node<v1::Divide>(scores, cap)),
                                        cap);
    // + additive mask, then softmax over the key axis.
    const auto masked = register_new_node<v1::Add>(capped, mask);
    const auto probs = register_new_node<ov::op::v8::Softmax>(masked, -1);
    return register_new_node<ov::op::v0::MatMul>(probs, V, false, false);
}

std::shared_ptr<ov::Node> ov::pass::PagedAttentionDecomposition::build_slot_indices(
    const ov::Output<ov::Node>& block_table_row,
    const ov::Output<ov::Node>& start_pos_scalar,
    const ov::Output<ov::Node>& count_scalar,
    const ov::Output<ov::Node>& block_size_scalar) {
    // Logical positions p = start_pos + [0, count), then slot(p) = block_table[p / block_size] * block_size +
    // p % block_size. All i32. The result is the flat row index into a [num_blocks * block_size, ...] cache.
    const auto zero_i32_s = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
    const auto one_i32_s = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {1}));
    const auto range = register_new_node<v4::Range>(zero_i32_s, count_scalar, one_i32_s, ov::element::i32);
    const auto p = register_new_node<v1::Add>(range, start_pos_scalar);  // [count]
    const auto logical_block = register_new_node<v1::Divide>(p, block_size_scalar);
    const auto slot_in_block = register_new_node<v1::Mod>(p, block_size_scalar);
    const auto block_number = register_new_node<v8::Gather>(block_table_row, logical_block, zero_i32_s);
    const auto base = register_new_node<v1::Multiply>(block_number, block_size_scalar);
    return register_new_node<v1::Add>(base, slot_in_block);
}

std::shared_ptr<ov::Node> ov::pass::PagedAttentionDecomposition::make_attention_mask(
    const ov::Output<ov::Node>& curr_seqlen_scalar,
    const ov::Output<ov::Node>& kv_len_scalar,
    const ov::Output<ov::Node>& past_seqlen_scalar,
    const ov::element::Type& compute_type,
    int64_t local_window_size) {
    const bool has_window = local_window_size >= 1;
    const auto zero_i64 = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
    const auto one_i64 = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    const auto zero_i32_s = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));
    const auto one_i32_s = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {1}));

    // Key positions [1, kv_len] (absolute key j) and query positions [curr, 1] (absolute query past + i).
    std::shared_ptr<ov::Node> hori =
        register_new_node<v4::Range>(zero_i32_s, kv_len_scalar, one_i32_s, ov::element::i32);
    hori = register_new_node<v0::Unsqueeze>(hori, zero_i64);
    std::shared_ptr<ov::Node> vert =
        register_new_node<v4::Range>(zero_i32_s, curr_seqlen_scalar, one_i32_s, ov::element::i32);
    vert = register_new_node<v0::Unsqueeze>(vert, one_i64);
    vert = register_new_node<v1::Add>(vert, past_seqlen_scalar);

    // Causal: mask key j when j > past + i. Optionally OR the sliding window: mask when (past + i) - j >= window.
    std::shared_ptr<ov::Node> masked = register_new_node<v1::Greater>(hori, vert);
    if (has_window) {
        const auto distance = register_new_node<v1::Subtract>(vert, hori);
        const auto window = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{}, {local_window_size}));
        const auto too_old = register_new_node<v1::GreaterEqual>(distance, window);
        masked = register_new_node<v1::LogicalOr>(masked, too_old);
    }

    // Additive mask: finite lowest() at masked positions (NaN-safe: no row is fully masked because every query
    // keeps its own diagonal key), 0 elsewhere. The magnitude must match the compute type so it does not
    // overflow to -inf when narrowed.
    const auto typed_zero = register_new_node(v0::Constant::create(compute_type, ov::Shape{}, {0}));
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

    return register_new_node<v1::Select>(masked, minus_inf, typed_zero);
}

std::shared_ptr<ov::Node> ov::pass::PagedAttentionDecomposition::get_dimensions(
    const std::shared_ptr<v3::ShapeOf>& shape,
    const std::vector<int>& dims) {
    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return register_new_node<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> ov::pass::PagedAttentionDecomposition::get_dimensions(const std::shared_ptr<ov::Node>& node,
                                                                                const std::vector<int>& dims) {
    return get_dimensions(register_new_node<v3::ShapeOf>(node), dims);
}

std::shared_ptr<ov::Node> ov::pass::PagedAttentionDecomposition::rotaryEmbedding(ov::Output<ov::Node> input,
                                                                                 ov::Output<ov::Node> cos,
                                                                                 ov::Output<ov::Node> sin,
                                                                                 bool interleaved) {
    const auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto unsqueeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    const auto cos_4d = register_new_node<v0::Unsqueeze>(cos, unsqueeze_axes);
    const auto sin_4d = register_new_node<v0::Unsqueeze>(sin, unsqueeze_axes);

    ov::Output<ov::Node> rope_input = input;
    std::shared_ptr<v3::ShapeOf> input_shape;
    std::shared_ptr<ov::Node> dim_bns, half_head_size;
    std::shared_ptr<v0::Constant> perm_5d;
    if (interleaved) {
        input_shape = register_new_node<v3::ShapeOf>(input);
        dim_bns = get_dimensions(input_shape, {0, 1, 2});
        half_head_size = get_dimensions(cos.get_node_shared_ptr(), {-1});
        perm_5d = v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});
        const auto deinterleave_5d = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, half_head_size, two}, 0);
        const auto reshaped_5d = register_new_node<v1::Reshape>(input, deinterleave_5d, false);
        const auto transposed_5d = register_new_node<v1::Transpose>(reshaped_5d, perm_5d);
        rope_input = register_new_node<v1::Reshape>(transposed_5d, input_shape, false);
    }

    const auto& cos_partial_shape = cos.get_partial_shape();
    const auto half_head_size_val =
        static_cast<int64_t>(cos_partial_shape[cos_partial_shape.rank().get_length() - 1].get_length());
    const auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    const auto split_lengths =
        v0::Constant::create(ov::element::i64, ov::Shape{2}, {half_head_size_val, half_head_size_val});
    const auto in_split = register_new_node<v1::VariadicSplit>(rope_input, split_axis, split_lengths)->outputs();
    const auto first_half_mul_cos = register_new_node<v1::Multiply>(in_split[0], cos_4d);
    const auto second_half_mul_sin = register_new_node<v1::Multiply>(in_split[1], sin_4d);
    const auto neg_one = v0::Constant::create(input.get_element_type(), ov::Shape{}, {-1.0f});
    const auto neg_second_sin = register_new_node<v1::Multiply>(second_half_mul_sin, neg_one);
    const auto res_0 = register_new_node<v1::Add>(first_half_mul_cos, neg_second_sin);
    const auto second_half_mul_cos = register_new_node<v1::Multiply>(in_split[1], cos_4d);
    const auto first_half_mul_sin = register_new_node<v1::Multiply>(in_split[0], sin_4d);
    const auto res_1 = register_new_node<v1::Add>(second_half_mul_cos, first_half_mul_sin);
    ov::Output<ov::Node> output = register_new_node<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);

    if (interleaved) {
        const auto reinterleave_5d = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, two, half_head_size}, 0);
        const auto result_5d = register_new_node<v1::Reshape>(output, reinterleave_5d, false);
        const auto result_transposed = register_new_node<v1::Transpose>(result_5d, perm_5d);
        output = register_new_node<v1::Reshape>(result_transposed, input_shape, false);
    }

    return output.get_node_shared_ptr();
}
