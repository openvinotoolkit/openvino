// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/group_query_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
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
#include "openvino/op/multiply.hpp"
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
    MATCHER_SCOPE(GroupQeuryAttentionDecomposition);
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
    const auto num_heads = node->get_num_heads();
    const auto kv_num_heads = node->get_kv_num_heads();
    const auto scale = node->get_scale();
    const auto do_rotary = node->get_do_rotary();
    const auto rotary_interleaved = node->get_rotary_interleaved();
    // TODO: add softcap support

    auto Q = node->input_value(0);
    auto K = node->input_value(1);
    auto V = node->input_value(2);
    auto past_key = node->input_value(3);
    auto past_value = node->input_value(4);
    auto seqlens_k = node->input_value(5);
    auto total_sequence_length = node->input_value(6);

    // Quantized KV cache (com.microsoft spec): past/present KV are i8/u8/f8e4m3 and are dequantized before the
    // attention math and (re)quantized when appended to the cache. Scales live at inputs 12 (K) / 13 (V).
    const bool kv_quantized = node->is_kv_quantized();
    const auto kv_cache_bit_width = node->get_kv_cache_bit_width();
    const auto k_quant_type = node->get_k_quant_type();
    const auto v_quant_type = node->get_v_quant_type();
    const auto kv_cache_type = past_key.get_element_type();
    ov::Output<ov::Node> k_scale, v_scale;
    if (kv_quantized) {
        k_scale = node->input_value(12);
        v_scale = node->input_value(13);
    }

    auto is_null = [](const ov::Output<ov::Node>& output) {
        return output.get_node_shared_ptr()->description() == "NullNode";
    };

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
        auto cos_cache = node->input_value(7);
        auto sin_cache = node->input_value(8);

        ov::Output<ov::Node> position_ids =
            register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
        if (node->get_input_size() > 9 && !is_null(node->input_value(9))) {
            // Flatten position_ids to 1D so that Gather produces 2D [seqlen, head_size/2] output,
            // ensuring correct 4D shapes after Unsqueeze in rotaryEmbedding.
            const auto neg_one = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}));
            position_ids = register_new_node<v1::Reshape>(node->input_value(9), neg_one, false);
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

    if (is_static_input) {
        // static design for GQA (KV cache is static max length, valid KVs are left align)
        // inputs are:
        //   1. past_key/past_value: [1, num_heads, max_seq_len, head_size], data is in the front along axis 2, [P0, P1,
        //   ..., Pn, 0, 0, ...]
        //   2. current K/V: [1, num_heads, current_kv_len, head_size], data is in the front along axis 2, [C0, C1, ...,
        //   Ck, 0, 0, ...]
        // Output present_key/present_value has the same shape with past_key/past_value, but with data in order [P0, P1,
        // ..., Pn, C0, C1, ..., Ck, 0, 0, ...]
        //
        // Use ScatterUpdate to scatter insert Current into Past
        // Insert current K/V at the correct position [past_seqlen, past_seqlen+curr_seqlen].
        std::shared_ptr<ov::Node> scatter_idx =
            register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
        scatter_idx = register_new_node<v1::Add>(scatter_idx, past_seqlen);
        const auto scatter_axis = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {2}));
        K = register_new_node<v3::ScatterUpdate>(past_key, scatter_idx, K, scatter_axis);
        V = register_new_node<v3::ScatterUpdate>(past_value, scatter_idx, V, scatter_axis);
    } else {
        auto construct_kv_cache = [&](const ov::Output<ov::Node>& past, const ov::Output<ov::Node>& current) {
            return register_new_node<v0::Concat>(ov::OutputVector{past, current}, 2);
        };
        past_key = register_new_node<v8::Slice>(past_key, zero, past_seqlen, one, two);
        past_value = register_new_node<v8::Slice>(past_value, zero, past_seqlen, one, two);
        K = construct_kv_cache(past_key, K);
        V = construct_kv_cache(past_value, V);
    }

    // present_key/present_value are the assembled cache (quantized when kv_quantized), matching the op outputs.
    ov::Output<ov::Node> present_k = K;
    ov::Output<ov::Node> present_v = V;

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

    // Make attention mask
    std::shared_ptr<ov::Node> mask;
    if (node->get_input_size() > 10 && !is_null(node->input_value(10))) {
        auto original_mask = node->input_value(10).get_node_shared_ptr();
        // Extract mask [num_heads, curr_seqlen, concat_kv_len] from 4D mask [1, num_heads, curr_seqlen, max_kv_len]
        auto axes_to_squeeze = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        auto mask_squeezed = register_new_node<v0::Squeeze>(original_mask, axes_to_squeeze);
        mask = register_new_node<v8::Slice>(mask_squeezed, zero, concat_kv_len, one, two);
    } else {
        std::shared_ptr<ov::Node> hori_range =
            register_new_node<v4::Range>(zero_without_shape, concat_kv_len_scalar, one_without_shape, ov::element::i64);
        hori_range = register_new_node<v0::Unsqueeze>(hori_range, zero);

        std::shared_ptr<ov::Node> vert_range =
            register_new_node<v4::Range>(zero_without_shape, curr_seqlen_scalar, one_without_shape, ov::element::i64);
        vert_range = register_new_node<v0::Unsqueeze>(vert_range, one);
        vert_range = register_new_node<v1::Add>(vert_range, past_seqlen);

        const auto triu = register_new_node<v1::Greater>(hori_range, vert_range);
        const auto typed_zero = register_new_node(v0::Constant::create(T, ov::Shape{}, {0}));
        // cf. make_attention_mask@src\plugins\intel_gpu\tests\common\subgraphs_builders.hpp
        // Mask with the type's finite lowest(), not -inf: a fully-masked row would otherwise softmax to 0/0 = NaN.
        std::shared_ptr<ov::Node> minus_inf = nullptr;
        if (T == ov::element::f32)
            minus_inf = register_new_node(v0::Constant::create(T, ov::Shape{}, {std::numeric_limits<float>::lowest()}));
        else if (T == ov::element::f16)
            minus_inf =
                register_new_node(v0::Constant::create(T, ov::Shape{}, {std::numeric_limits<ov::float16>::lowest()}));
        mask = register_new_node<v1::Select>(triu, minus_inf, typed_zero);
    }

    std::shared_ptr<ov::Node> qga_output;
    if (scale != 0.0f) {
        auto scale_node = register_new_node(v0::Constant::create(T, Shape{}, {scale}));
        qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, scale_node, false);
    } else {
        qga_output = register_new_node<v13::ScaledDotProductAttention>(Q, K, V, mask, false);
    }

    // transpose the result from (batch_size, num_heads, sequence_length, head_size)
    // to (batch_size, sequence_length, num_heads * head_size)
    auto perm = register_new_node(v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
    auto qga_output_transposed = register_new_node<v1::Transpose>(qga_output, perm);
    auto dim_merge_shape = register_new_node(v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1}));
    auto output = register_new_node<v1::Reshape>(qga_output_transposed, dim_merge_shape, true)->output(0);

    return {output, present_k, present_v};
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

    // Unsqueeze cos/sin to 4D [1, 1, seqlen, head_size/2] to match RoPE fusion pattern
    auto unsqueeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto cos_4d = register_new_node<v0::Unsqueeze>(cos, unsqueeze_axes);
    auto sin_4d = register_new_node<v0::Unsqueeze>(sin, unsqueeze_axes);

    // For interleaved mode, deinterleave first so the core RoPE formula is identical
    ov::Output<ov::Node> rope_input = input;
    std::shared_ptr<v3::ShapeOf> input_shape;
    std::shared_ptr<ov::Node> dim_bns, half_head_size;
    std::shared_ptr<v0::Constant> perm_5d;
    if (interleaved) {
        input_shape = register_new_node<v3::ShapeOf>(input);
        dim_bns = get_dimensions(input_shape, {0, 1, 2});
        half_head_size = get_dimensions(cos.get_node_shared_ptr(), {-1});
        perm_5d = v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});

        // Deinterleave: [bs,nh,seq,head_size]
        //   -> reshape [bs,nh,seq,head_size/2,2]
        //   -> transpose [bs,nh,seq,2,head_size/2]
        //   -> reshape [bs,nh,seq,head_size]  (now [first_half, second_half])
        auto deinterleave_5d = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, half_head_size, two}, 0);
        auto reshaped_5d = register_new_node<v1::Reshape>(input, deinterleave_5d, false);
        auto transposed_5d = register_new_node<v1::Transpose>(reshaped_5d, perm_5d);
        rope_input = register_new_node<v1::Reshape>(transposed_5d, input_shape, false);
    }

    // Core RoPE formula (matches RoPEFusionGPTOSS pattern for both modes)
    // first_ = first_half * cos - second_half * sin
    // second_ = second_half * cos + first_half * sin
    const auto& cos_partial_shape = cos.get_partial_shape();
    const auto half_head_size_val =
        static_cast<int64_t>(cos_partial_shape[cos_partial_shape.rank().get_length() - 1].get_length());
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
        // Re-interleave: [bs,nh,seq,head_size]
        //   -> reshape [bs,nh,seq,2,head_size/2]
        //   -> transpose [bs,nh,seq,head_size/2,2]
        //   -> reshape [bs,nh,seq,head_size]
        auto reinterleave_5d = register_new_node<v0::Concat>(ov::NodeVector{dim_bns, two, half_head_size}, 0);
        auto result_5d = register_new_node<v1::Reshape>(output, reinterleave_5d, false);
        auto result_transposed = register_new_node<v1::Transpose>(result_5d, perm_5d);
        output = register_new_node<v1::Reshape>(result_transposed, input_shape, false);
    }

    return output.get_node_shared_ptr();
}

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::make_kv_scale(const ov::Output<ov::Node>& scale,
                                                                                    int64_t kv_num_heads,
                                                                                    const std::string& quant_type) {
    // The KV cache is laid out as [batch, kv_num_heads, seq_len, head_size]. Reshape the flat scale so it
    // broadcasts along that layout. A fully static target shape is used (no -1 wildcard) so the result stays
    // static-shaped for plugins (e.g. NPU) that require static shapes. A fresh shape Constant is built per
    // call so no two GQA layers alias it.
    std::vector<int64_t> target_shape;
    if (quant_type == "PER_CHANNEL") {
        // Per-channel scale has kv_num_heads * head_size elements, head-major (scale[kv_head * head_size + ch]).
        // Reshape to [1, kv_num_heads, 1, head_size] to broadcast over batch and seq. head_size is derived from
        // the (static) scale length when known, otherwise falls back to a -1 wildcard.
        int64_t head_size = -1;
        const auto& scale_pshape = scale.get_partial_shape();
        if (scale_pshape.rank().is_static() && scale_pshape.rank().get_length() == 1 && scale_pshape[0].is_static()) {
            head_size = scale_pshape[0].get_length() / kv_num_heads;
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
    const std::string& quant_type,
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

std::shared_ptr<ov::Node> ov::pass::GroupQueryAttentionDecomposition::quantize_kv(const ov::Output<ov::Node>& current,
                                                                                  const ov::Output<ov::Node>& scale,
                                                                                  int64_t kv_num_heads,
                                                                                  int64_t kv_cache_bit_width,
                                                                                  const std::string& quant_type,
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
