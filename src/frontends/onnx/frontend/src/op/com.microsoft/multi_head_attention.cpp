// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils/attention.hpp"
#include "utils/common.hpp"
#include "utils/split.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace detail {
namespace {

using ov::frontend::onnx::attention::get_dimensions;

// Reshape 3D input (batch, seq, num_heads * head_size) to 4D (batch, num_heads, seq, head_size)
ov::Output<ov::Node> reshape_3d_to_4d(const ov::Output<ov::Node>& input, int64_t num_heads) {
    // Reshape to (batch, seq, num_heads, head_size)
    auto reshape_pattern = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0ll, 0ll, num_heads, -1ll});
    auto reshaped = std::make_shared<v1::Reshape>(input, reshape_pattern, true);
    // Transpose to (batch, num_heads, seq, head_size)
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    return std::make_shared<v1::Transpose>(reshaped, perm);
}

// Reshape 4D output (batch, num_heads, seq, head_size) back to 3D (batch, seq, num_heads * head_size)
ov::Output<ov::Node> reshape_4d_to_3d(const ov::Output<ov::Node>& output) {
    // Transpose from (batch, num_heads, seq, head_size) to (batch, seq, num_heads, head_size)
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto transposed = std::make_shared<v1::Transpose>(output, perm);
    // Reshape to (batch, seq, num_heads * head_size)
    auto reshape_pattern = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, -1});
    return std::make_shared<v1::Reshape>(transposed, reshape_pattern, true);
}

// Split bias into Q, K, V parts and reshape each to (1, num_heads, 1, head_size) for broadcasting.
OutputVector split_bias(const Output<ov::Node>& bias,
                        const Output<ov::Node>& Q,
                        const Output<ov::Node>& K,
                        const Output<ov::Node>& V,
                        int64_t num_heads) {
    auto Q_shape = std::make_shared<v3::ShapeOf>(Q);
    auto K_shape = std::make_shared<v3::ShapeOf>(K);
    auto V_shape = std::make_shared<v3::ShapeOf>(V);
    auto Q_head_depth = get_dimensions(Q_shape, {3});
    auto K_head_depth = get_dimensions(K_shape, {3});
    auto V_head_depth = get_dimensions(V_shape, {3});
    auto num_heads_node = ov::op::v0::Constant::create(ov::element::i64, Shape{1}, {num_heads});
    auto Q_hidden_dim = std::make_shared<v1::Multiply>(Q_head_depth, num_heads_node);
    auto K_hidden_dim = std::make_shared<v1::Multiply>(K_head_depth, num_heads_node);
    auto V_hidden_dim = std::make_shared<v1::Multiply>(V_head_depth, num_heads_node);

    auto axis_node = v0::Constant::create(ov::element::i64, Shape{}, {0});
    auto split_lengths_node = std::make_shared<v0::Concat>(ov::NodeVector{Q_hidden_dim, K_hidden_dim, V_hidden_dim}, 0);
    auto variadic_split = std::make_shared<v1::VariadicSplit>(bias, axis_node, split_lengths_node);

    auto outputs = variadic_split->outputs();
    auto reshape_pattern = v0::Constant::create(ov::element::i64, ov::Shape{4}, {1ll, num_heads, 1ll, -1ll});
    outputs[0] = std::make_shared<v1::Reshape>(outputs[0], reshape_pattern, false);
    outputs[1] = std::make_shared<v1::Reshape>(outputs[1], reshape_pattern, false);
    outputs[2] = std::make_shared<v1::Reshape>(outputs[2], reshape_pattern, false);
    return outputs;
}

// Convert boolean mask to float additive mask: true -> 0.0, false -> mask_filter_value
ov::Output<ov::Node> convert_boolean_mask(const ov::Output<ov::Node>& mask,
                                          const ov::element::Type& type,
                                          float mask_filter_value = -10000.0f) {
    auto zero = v0::Constant::create(type, ov::Shape{}, {0.0f});
    auto neg_large = v0::Constant::create(type, ov::Shape{}, {mask_filter_value});
    return std::make_shared<v1::Select>(mask, zero, neg_large);
}

// Build additive unidirectional mask of shape (seq_q, seq_kv): 0 for allowed, -10000 for masked.
// When use_offset=true, accounts for KV cache offset so that query position i attends to
// key positions j where j <= i + (seq_kv - seq_q). Use this for KV cache scenarios.
// When use_offset=false, builds a simple lower-triangular mask matching np.tril(k=0),
// where query position i attends to key positions j where j <= i.
ov::Output<ov::Node> build_unidirectional_mask(const ov::Output<ov::Node>& Q,
                                               const ov::Output<ov::Node>& K,
                                               bool use_offset) {
    auto q_shape = std::make_shared<v3::ShapeOf>(Q);
    auto k_shape = std::make_shared<v3::ShapeOf>(K);
    // Q is 4D: (B, heads, seq_q, head_size), K is 4D: (B, heads, seq_kv, head_size)
    auto seq_q = get_dimensions(q_shape, {2});
    auto seq_kv = get_dimensions(k_shape, {2});

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    auto seq_q_scalar = std::make_shared<v0::Squeeze>(seq_q, zero);
    auto seq_kv_scalar = std::make_shared<v0::Squeeze>(seq_kv, zero);

    // Column indices: [0, 1, ..., seq_kv-1]
    auto col_indices = std::make_shared<v4::Range>(zero, seq_kv_scalar, one, ov::element::i64);

    std::shared_ptr<ov::Node> row_indices;
    if (use_offset) {
        // Row indices adjusted for KV cache offset: [offset, offset+1, ..., offset+seq_q-1]
        auto offset = std::make_shared<v1::Subtract>(seq_kv_scalar, seq_q_scalar);
        auto end = std::make_shared<v1::Add>(offset, seq_q_scalar);
        row_indices = std::make_shared<v4::Range>(offset, end, one, ov::element::i64);
    } else {
        // Row indices without offset: [0, 1, ..., seq_q-1] (matches np.tril(k=0))
        row_indices = std::make_shared<v4::Range>(zero, seq_q_scalar, one, ov::element::i64);
    }

    // Unsqueeze rows to (seq_q, 1) for broadcasting with (seq_kv,)
    auto axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto rows_2d = std::make_shared<v0::Unsqueeze>(row_indices, axis);

    // Lower-triangular: position (i, j) allowed when adjusted_row[i] >= col[j]
    auto is_allowed = std::make_shared<v1::GreaterEqual>(rows_2d, col_indices);

    // Convert boolean to additive float mask: true -> 0.0, false -> -10000.0
    return convert_boolean_mask(is_allowed, Q.get_element_type());
}

// Build 4D attention mask from key_padding_mask input, which can have rank 1, 2 or 3.
ov::Output<ov::Node> build_mask(const ov::Output<ov::Node>& key_padding_mask,
                                const ov::Output<ov::Node>& Q,
                                const ov::Output<ov::Node>& K,
                                float mask_filter_value) {
    auto q_shape = std::make_shared<v3::ShapeOf>(Q);
    auto k_shape = std::make_shared<v3::ShapeOf>(K);
    // Q is 4D: (B, heads, seq_q, head_size), K is 4D: (B, heads, seq_kv, head_size)
    auto seq_q = get_dimensions(q_shape, {2});
    auto seq_kv = get_dimensions(k_shape, {2});
    auto batch = get_dimensions(q_shape, {0});

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto zero_1d = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto one_1d = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});

    auto seq_q_scalar = std::make_shared<v0::Squeeze>(seq_q, zero);
    auto seq_kv_scalar = std::make_shared<v0::Squeeze>(seq_kv, zero);

    auto key_padding_rank = key_padding_mask.get_partial_shape().rank();
    FRONT_END_GENERAL_CHECK(key_padding_rank.is_static(), "'key_padding_mask' rank must be static");
    ov::Output<ov::Node> output_mask_3d;
    if (key_padding_rank.get_length() == 1) {
        // Column indices: [0, 1, ..., seq_kv-1]
        auto col_indices = std::make_shared<v4::Range>(zero, seq_kv_scalar, one, ov::element::i64);

        // Take only first B elements, in case of packed 3*B+2 mask, still only first B needed
        auto sliced = std::make_shared<v8::Slice>(key_padding_mask, zero_1d, batch, one_1d, zero_1d);
        // Unsqueeze mask to (B, 1)
        auto converted = std::make_shared<v0::Convert>(sliced, ov::element::i64);
        auto batch_indices = std::make_shared<v0::Unsqueeze>(converted, one_1d);

        auto is_allowed = std::make_shared<v1::Greater>(batch_indices, col_indices);

        auto mask_3d = std::make_shared<v0::Unsqueeze>(is_allowed, one_1d);  // (B, 1, seq_kv)

        auto broadcast_shape = std::make_shared<v0::Concat>(ov::NodeVector{batch, seq_q, seq_kv}, 0);
        output_mask_3d = std::make_shared<v1::Broadcast>(mask_3d, broadcast_shape);
    } else if (key_padding_rank.get_length() == 2) {
        auto mask_3d = std::make_shared<v0::Unsqueeze>(key_padding_mask, one_1d);  // (B, 1, seq_kv)
        auto converted = std::make_shared<v0::Convert>(mask_3d, ov::element::boolean);
        auto broadcast_shape = std::make_shared<v0::Concat>(ov::NodeVector{batch, seq_q, seq_kv}, 0);
        output_mask_3d = std::make_shared<v1::Broadcast>(converted, broadcast_shape);
    } else {
        FRONT_END_GENERAL_CHECK(key_padding_rank.get_length() == 3,
                                "Expected 'key_padding_mask' to have a rank of 1, 2 or 3, got: ",
                                key_padding_rank.get_length());
        output_mask_3d = std::make_shared<v0::Convert>(key_padding_mask, ov::element::boolean);
    }
    // Convert boolean to additive float mask: true -> 0.0, false -> -10000.0
    auto output_mask_4d = std::make_shared<v0::Unsqueeze>(output_mask_3d, one_1d);  // (B, 1, seq_q, seq_kv)
    return convert_boolean_mask(output_mask_4d, Q.get_element_type(), mask_filter_value);
}

// Handle different input formats for Q, K, V: separate vs packed. Returns {Q, K, V} in (batch, num_heads, seq_len,
// head_size) and is_regular_QKV flag.
std::tuple<ov::OutputVector, bool> prepare_qkv(const ov::frontend::onnx::Node& node,
                                               const ov::OutputVector& inputs,
                                               int64_t num_heads) {
    ov::Output<ov::Node> Q, K, V;
    bool is_packed_QKV = !common::is_input_valid(node, 1) && !common::is_input_valid(node, 2);
    bool is_packed_KV = !common::is_input_valid(node, 2) && !is_packed_QKV;
    bool is_cross_attn = false;

    if (is_packed_QKV) {
        // Packed QKV: split input 0 into Q, K, V
        // In this case input[0] should have shape (batch_size, kv_sequence_length, num_heads, 3, head_size)
        auto split = ov::op::util::make_split(inputs[0], 3, 3);
        auto reshape_pattern = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, -1});
        auto reshaped_Q = std::make_shared<v1::Reshape>(split[0], reshape_pattern, true);
        auto reshaped_K = std::make_shared<v1::Reshape>(split[1], reshape_pattern, true);
        auto reshaped_V = std::make_shared<v1::Reshape>(split[2], reshape_pattern, true);
        auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        Q = std::make_shared<v1::Transpose>(reshaped_Q, perm);
        K = std::make_shared<v1::Transpose>(reshaped_K, perm);
        V = std::make_shared<v1::Transpose>(reshaped_V, perm);
    } else if (is_packed_KV) {
        // Packed KV: input 1 is K/V concatenated; split into K and V
        // Q should be 3D in this case
        Q = reshape_3d_to_4d(inputs[0], num_heads);
        // In this case input[1] should have shape (batch_size, kv_sequence_length, num_heads, 2, head_size)
        auto split = ov::op::util::make_split(inputs[1], 2, 3);
        auto reshape_pattern = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 0, 0, -1});
        auto reshaped_K = std::make_shared<v1::Reshape>(split[0], reshape_pattern, true);
        auto reshaped_V = std::make_shared<v1::Reshape>(split[1], reshape_pattern, true);
        auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
        K = std::make_shared<v1::Transpose>(reshaped_K, perm);
        V = std::make_shared<v1::Transpose>(reshaped_V, perm);
    } else {
        Q = reshape_3d_to_4d(inputs[0], num_heads);

        K = inputs[1];
        V = inputs[2];
        // Check if this is cross-attention: K and V are 4D in this case
        auto k_rank = K.get_partial_shape().rank();
        auto v_rank = V.get_partial_shape().rank();
        if (k_rank.is_static() && v_rank.is_static()) {
            bool kv_is_3d = (k_rank.get_length() == 3) && (v_rank.get_length() == 3);
            is_cross_attn = (k_rank.get_length() == 4) && (v_rank.get_length() == 4);
            CHECK_VALID_NODE(node,
                             kv_is_3d || is_cross_attn,
                             "KV input rank must be 3 or 4, got: ",
                             k_rank.get_length(),
                             v_rank.get_length());
        }
        if (!is_cross_attn) {
            // For self-attention, reshape K and V to 4D
            K = reshape_3d_to_4d(K, num_heads);
            V = reshape_3d_to_4d(V, num_heads);
        }
    }
    bool is_regular_QKV = !(is_packed_QKV || is_packed_KV || is_cross_attn);
    return {{Q, K, V}, is_regular_QKV};
}

// Apply beam search cache indirection to reorder cached K/V by beam indices.
ov::OutputVector apply_cache_indirection(const ov::Output<ov::Node>& cache_K,
                                         const ov::Output<ov::Node>& cache_V,
                                         const ov::Output<ov::Node>& indirection_input,
                                         const ov::Output<ov::Node>& past_seq_len_1d) {
    auto zero_1d = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    auto one_1d = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axis_2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    ov::Output<ov::Node> cache_indirection =
        std::make_shared<v8::Slice>(indirection_input, zero_1d, past_seq_len_1d, one_1d, axis_2);
    auto indirection_shape = std::make_shared<v3::ShapeOf>(cache_indirection);
    auto cache_shape = std::make_shared<v3::ShapeOf>(cache_K);
    auto num_beams = get_dimensions(indirection_shape, {1});
    auto indirection_batch = get_dimensions(indirection_shape, {0});
    auto cache_num_heads = get_dimensions(cache_shape, {1});
    auto cache_head_size = get_dimensions(cache_shape, {3});
    auto reshape_pattern = std::make_shared<v0::Concat>(
        ov::OutputVector{indirection_batch, num_beams, cache_num_heads, past_seq_len_1d, cache_head_size},
        0);
    ov::Output<ov::Node> cache_K_out = std::make_shared<v1::Reshape>(cache_K, reshape_pattern, false);
    ov::Output<ov::Node> cache_V_out = std::make_shared<v1::Reshape>(cache_V, reshape_pattern, false);

    auto indirection_reshape_pattern =
        std::make_shared<v0::Concat>(ov::OutputVector{indirection_batch, num_beams, one_1d, past_seq_len_1d, one_1d},
                                     0);
    cache_indirection = std::make_shared<v1::Reshape>(cache_indirection, indirection_reshape_pattern, false);
    cache_indirection = std::make_shared<v1::Broadcast>(cache_indirection, reshape_pattern);

    // Gather with indirection: (B * beam_num, seq_kv)
    cache_K_out = std::make_shared<v6::GatherElements>(cache_K_out, cache_indirection, 1);
    cache_V_out = std::make_shared<v6::GatherElements>(cache_V_out, cache_indirection, 1);

    cache_K_out = std::make_shared<v1::Reshape>(cache_K_out, cache_shape, false);
    cache_V_out = std::make_shared<v1::Reshape>(cache_V_out, cache_shape, false);
    return {cache_K_out, cache_V_out};
}

// Handle KV cache: buffer sharing, cache indirection, and concatenation with current K/V.
// Modifies K, V by prepending cached values. Returns {present_key, present_value}.
ov::OutputVector apply_kv_cache(const ov::Output<ov::Node>& K,
                                const ov::Output<ov::Node>& V,
                                const ov::OutputVector& inputs,
                                bool has_buffer_sharing,
                                bool has_cache_indirection) {
    ov::Output<ov::Node> present_key, present_value;
    ov::Output<ov::Node> cache_K = inputs[6];
    ov::Output<ov::Node> cache_V = inputs[7];
    if (has_buffer_sharing) {
        auto past_seq_len_converted = std::make_shared<v0::Convert>(inputs[8], ov::element::i64);
        auto new_shape = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto past_seq_len_1d = std::make_shared<v1::Reshape>(past_seq_len_converted, new_shape, false);

        auto k_shape = std::make_shared<v3::ShapeOf>(K);
        auto seq_kv = get_dimensions(k_shape, {2});

        // First, update the present values
        auto start = std::make_shared<v0::Squeeze>(past_seq_len_1d);
        auto end = std::make_shared<v0::Squeeze>(std::make_shared<v1::Add>(past_seq_len_1d, seq_kv));
        auto step = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto axis_2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto indices = std::make_shared<v4::Range>(start, end, step, ov::element::i64);
        present_key = std::make_shared<v3::ScatterUpdate>(cache_K, indices, K, axis_2);
        present_value = std::make_shared<v3::ScatterUpdate>(cache_V, indices, V, axis_2);

        // Then slice the past values
        auto zero_1d = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto one_1d = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        cache_K = std::make_shared<v8::Slice>(cache_K, zero_1d, past_seq_len_1d, one_1d, axis_2);
        cache_V = std::make_shared<v8::Slice>(cache_V, zero_1d, past_seq_len_1d, one_1d, axis_2);
        if (has_cache_indirection) {
            auto kv_cache_with_indirection = apply_cache_indirection(cache_K, cache_V, inputs[9], past_seq_len_1d);
            cache_K = kv_cache_with_indirection[0];
            cache_V = kv_cache_with_indirection[1];
        }
    }

    auto K_out = std::make_shared<v0::Concat>(ov::OutputVector{cache_K, K}, 2);
    auto V_out = std::make_shared<v0::Concat>(ov::OutputVector{cache_V, V}, 2);
    // If buffer sharing is disabled, present_key and present_value are K and V after concatenation
    if (!has_buffer_sharing) {
        present_key = K_out;
        present_value = V_out;
    }
    return {K_out, V_out, present_key, present_value};
}

// Build SDPA-based attention (primary fast path)
ov::Output<ov::Node> build_sdpa(const ov::Output<ov::Node>& Q,
                                const ov::Output<ov::Node>& K,
                                const ov::Output<ov::Node>& V,
                                bool has_mask,
                                const ov::Output<ov::Node>& attn_mask,
                                float scale_attr) {
    ov::OutputVector inputs{Q, K, V};
    if (has_mask) {
        inputs.push_back(attn_mask);
    }
    if (scale_attr != 0.0f) {
        if (!has_mask) {
            // SDPA interprets inputs positionally (index 3 = mask, index 4 = scale),
            // so a zero mask placeholder is needed when only scale is provided
            inputs.push_back(v0::Constant::create(Q.get_element_type(), ov::Shape{}, {0.0f}));
        }
        inputs.push_back(v0::Constant::create(Q.get_element_type(), ov::Shape{}, {scale_attr}));
    }
    return std::make_shared<v13::ScaledDotProductAttention>(inputs, false)->output(0);
}

// Build manual attention decomposition (for qk_matmul_output only)
// Returns {Y, qk_matmul_output}
ov::OutputVector build_manual_attention(const ov::Output<ov::Node>& Q,
                                        const ov::Output<ov::Node>& K,
                                        const ov::Output<ov::Node>& V,
                                        bool has_mask,
                                        const ov::Output<ov::Node>& attn_mask,
                                        float scale_attr) {
    // 1. Q @ K^T
    auto qk = std::make_shared<v0::MatMul>(Q, K, false, true);

    // 2. Apply scale
    std::shared_ptr<ov::Node> scaled_qk;
    if (scale_attr != 0.0f) {
        auto scale_node = v0::Constant::create(Q.get_element_type(), ov::Shape{}, {scale_attr});
        scaled_qk = std::make_shared<v1::Multiply>(qk, scale_node);
    } else {
        // Default scale: 1/sqrt(head_size). Q is always 4D here: (B, heads, seq, head_size)
        auto q_shape = std::make_shared<v3::ShapeOf>(Q);
        auto head_size = get_dimensions(q_shape, {3});
        auto head_size_f = std::make_shared<v0::Convert>(head_size, Q.get_element_type());
        auto sqrt_head = std::make_shared<v0::Sqrt>(head_size_f);
        scaled_qk = std::make_shared<v1::Divide>(qk, sqrt_head);
    }

    // 3. Apply attention mask
    std::shared_ptr<ov::Node> masked = scaled_qk;
    if (has_mask) {
        masked = std::make_shared<v1::Add>(scaled_qk, attn_mask);
    }

    // Capture qk_matmul_output after mask
    auto qk_debug_output = masked->output(0);

    // 5. Softmax
    auto softmax_out = std::make_shared<v8::Softmax>(masked, -1);

    // 6. softmax @ V
    auto output = std::make_shared<v0::MatMul>(softmax_out, V);

    return {output->output(0), qk_debug_output};
}

}  // namespace
}  // namespace detail

namespace opset_1 {
ov::OutputVector multi_head_attention(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();
    CHECK_VALID_NODE(node, num_inputs >= 1, "MultiHeadAttention expects at least 1 input, got: ", num_inputs);

    // Attributes
    int64_t num_heads = node.get_attribute_value<int64_t>("num_heads");
    float mask_filter_value = node.get_attribute_value<float>("mask_filter_value", -10000.0f);
    bool unidirectional = static_cast<bool>(node.get_attribute_value<int64_t>("unidirectional", 0));
    float scale_attr = node.get_attribute_value<float>("scale", 0);

    // Required inputs
    CHECK_VALID_NODE(node, num_heads > 0, "num_heads attribute should be > 0");
    auto [qkv, is_regular_QKV] = detail::prepare_qkv(node, inputs, num_heads);
    auto Q = qkv[0], K = qkv[1], V = qkv[2];

    // Optional inputs
    bool has_bias = common::is_input_valid(node, 3);
    bool has_key_padding_mask = common::is_input_valid(node, 4);
    bool has_attention_bias = common::is_input_valid(node, 5);
    bool has_past_key = common::is_input_valid(node, 6);
    bool has_past_value = common::is_input_valid(node, 7);
    bool has_buffer_sharing = common::is_input_valid(node, 8);
    bool has_cache_indirection = common::is_input_valid(node, 9);

    CHECK_VALID_NODE(node,
                     !has_cache_indirection || has_buffer_sharing,
                     "cache_indirection is only supported in buffer_sharing mode");

    CHECK_VALID_NODE(node,
                     !has_buffer_sharing || has_past_key,
                     "buffer_sharing is supported when past key/value are present");

    CHECK_VALID_NODE(node,
                     has_past_key == has_past_value,
                     "past_key and past_value must be both present or both absent");
    CHECK_VALID_NODE(node,
                     !has_past_key || is_regular_QKV,
                     "past_key and past_value are only supported in unpacked 3D case");

    if (has_bias) {
        auto bias_splits = detail::split_bias(inputs[3], Q, K, V, num_heads);
        Q = std::make_shared<v1::Add>(Q, bias_splits[0]);
        K = std::make_shared<v1::Add>(K, bias_splits[1]);
        V = std::make_shared<v1::Add>(V, bias_splits[2]);
    }

    // Handle KV cache
    ov::Output<ov::Node> present_key = K;
    ov::Output<ov::Node> present_value = V;
    if (has_past_key) {
        auto kv_cache_result = detail::apply_kv_cache(K, V, inputs, has_buffer_sharing, has_cache_indirection);
        K = kv_cache_result[0];
        V = kv_cache_result[1];
        present_key = kv_cache_result[2];
        present_value = kv_cache_result[3];
    }

    // Prepare attention mask
    ov::Output<ov::Node> attn_mask;
    if (has_key_padding_mask) {
        attn_mask = inputs[4];
        auto attn_rank = attn_mask.get_partial_shape().rank();
        CHECK_VALID_NODE(node, attn_rank.is_static(), "Attention rank must be static, got rank: ", attn_rank);

        attn_mask = detail::build_mask(attn_mask, Q, K, mask_filter_value);
    }

    // Build an explicit unidirectional mask instead of relying on SDPA's internal is_causal flag.
    // SDPA's internal mask uses offset-based semantics (ncausal = kv_len - q_len + m + 1)
    // which doesn't match the ONNX spec's np.tril(k=0) for non-square attention matrices
    // (seq_q != seq_kv). For KV cache scenarios, use offset to account for past sequence.
    if (unidirectional) {
        auto unidirectional_mask = detail::build_unidirectional_mask(Q, K, has_past_key);
        if (has_key_padding_mask) {
            attn_mask = std::make_shared<v1::Add>(attn_mask, unidirectional_mask);
        } else {
            attn_mask = unidirectional_mask;
        }
        has_key_padding_mask = true;
    }

    if (has_attention_bias) {
        auto attention_bias = inputs[5];
        auto bias_rank = attention_bias.get_partial_shape().rank();
        CHECK_VALID_NODE(node,
                         bias_rank.is_static() && bias_rank.get_length() == 4,
                         "attention_bias must have rank 4, got: ",
                         bias_rank);
        if (has_key_padding_mask) {
            attn_mask = std::make_shared<v1::Add>(attn_mask, attention_bias);
        } else {
            attn_mask = attention_bias;
        }
        has_key_padding_mask = true;
    }

    // Choose execution path
    ov::Output<ov::Node> Y;
    ov::Output<ov::Node> qk_debug_output;

    // Determine number of requested outputs
    size_t num_outputs = node.get_outputs_size();
    const auto& output_names = node.get_output_names();
    bool needs_qk_output = output_names.size() > 3 && !output_names[3].get().empty();

    if (needs_qk_output) {
        // Manual decomposition path (softcap or debug output)
        auto results = detail::build_manual_attention(Q, K, V, has_key_padding_mask, attn_mask, scale_attr);
        Y = results[0];
        qk_debug_output = results[1];
    } else {
        // SDPA path (primary fast path)
        Y = detail::build_sdpa(Q, K, V, has_key_padding_mask, attn_mask, scale_attr);
    }

    Y = detail::reshape_4d_to_3d(Y);

    // Build output vector.
    // Output names from the ONNX graph determine which outputs are actually requested.
    // Empty names indicate unused optional outputs — push NullNode for those to avoid
    // creating shared input/output parameters that confuse port resolution.
    ov::OutputVector results;
    results.push_back(Y);

    if (num_outputs > 1) {
        if (!output_names[1].get().empty()) {
            results.push_back(present_key);
        } else {
            results.push_back(std::make_shared<NullNode>()->output(0));
        }
    }
    if (num_outputs > 2) {
        if (!output_names[2].get().empty()) {
            results.push_back(present_value);
        } else {
            results.push_back(std::make_shared<NullNode>()->output(0));
        }
    }
    if (num_outputs > 3) {
        if (qk_debug_output.get_node() && !output_names[3].get().empty()) {
            results.push_back(qk_debug_output);
        } else {
            results.push_back(std::make_shared<NullNode>()->output(0));
        }
    }

    return results;
}

ONNX_OP("MultiHeadAttention", OPSET_SINCE(1), com_microsoft::opset_1::multi_head_attention, MICROSOFT_DOMAIN);
}  // namespace opset_1

}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
