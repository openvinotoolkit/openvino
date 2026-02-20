// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/attention.hpp"

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace detail {
namespace {

using ov::frontend::onnx::attention::get_dimensions;

// Reshape 3D input (batch, seq, num_heads * head_size) to 4D (batch, num_heads, seq, head_size)
// Uses direct reshape matching the ONNX Attention spec (no transpose):
// the hidden dimension is split as [num_heads, seq, head_size] in row-major order.
ov::Output<ov::Node> reshape_3d_to_4d(const ov::Output<ov::Node>& input, int64_t num_heads) {
    auto input_shape = std::make_shared<v3::ShapeOf>(input);
    auto batch = get_dimensions(input_shape, {0});
    auto seq = get_dimensions(input_shape, {1});
    auto num_heads_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {num_heads});
    auto hidden = get_dimensions(input_shape, {2});
    auto head_size = std::make_shared<v1::Divide>(hidden, num_heads_node);
    // Direct reshape to (batch, num_heads, seq, head_size)
    auto new_shape = std::make_shared<v0::Concat>(ov::NodeVector{batch, num_heads_node, seq, head_size}, 0);
    return std::make_shared<v1::Reshape>(input, new_shape, false);
}

// Reshape 4D output (batch, num_heads, seq, head_size) back to 3D (batch, seq, num_heads * head_size)
ov::Output<ov::Node> reshape_4d_to_3d(const ov::Output<ov::Node>& output) {
    // Transpose from (batch, num_heads, seq, head_size) to (batch, seq, num_heads, head_size)
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto transposed = std::make_shared<v1::Transpose>(output, perm);
    // Reshape to (batch, seq, num_heads * head_size)
    auto reshape_pattern = v0::Constant::create(ov::element::i32, ov::Shape{3}, {0, 0, -1});
    return std::make_shared<v1::Reshape>(transposed, reshape_pattern, true);
}

// Repeat K/V heads for GQA: (B, kv_h, seq, h) -> (B, kv_h * n_rep, seq, h)
// Uses Tile to match ONNX spec's np.tile(K, [1, n_rep, 1, 1]) behavior.
// n_rep = q_num_heads / kv_num_heads
ov::Output<ov::Node> repeat_kv(const ov::Output<ov::Node>& input, int64_t n_rep) {
    if (n_rep == 1) {
        return input;
    }
    auto repeats = v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{1, n_rep, 1, 1});
    return std::make_shared<v0::Tile>(input, repeats);
}

// Convert boolean mask to float additive mask: true -> 0.0, false -> -10000.0
ov::Output<ov::Node> convert_boolean_mask(const ov::Output<ov::Node>& mask, const ov::element::Type& type) {
    auto zero = v0::Constant::create(type, ov::Shape{}, {0.0f});
    auto neg_large = v0::Constant::create(type, ov::Shape{}, {-10000.0f});
    return std::make_shared<v1::Select>(mask, zero, neg_large);
}

// Build additive causal mask of shape (seq_q, seq_kv): 0 for allowed, -10000 for masked.
// Accounts for KV cache offset so that query position i attends to key positions j where j <= i + (S - L).
ov::Output<ov::Node> build_causal_mask(const ov::Output<ov::Node>& Q, const ov::Output<ov::Node>& K) {
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

    // Row indices adjusted for KV cache offset: [offset, offset+1, ..., offset+seq_q-1]
    auto offset = std::make_shared<v1::Subtract>(seq_kv_scalar, seq_q_scalar);
    auto end = std::make_shared<v1::Add>(offset, seq_q_scalar);
    auto row_indices = std::make_shared<v4::Range>(offset, end, one, ov::element::i64);

    // Unsqueeze rows to (seq_q, 1) for broadcasting with (seq_kv,)
    auto axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto rows_2d = std::make_shared<v0::Unsqueeze>(row_indices, axis);

    // Lower-triangular: position (i, j) allowed when adjusted_row[i] >= col[j]
    auto is_allowed = std::make_shared<v1::GreaterEqual>(rows_2d, col_indices);

    // Convert boolean to additive float mask: true -> 0.0, false -> -10000.0
    return convert_boolean_mask(is_allowed, Q.get_element_type());
}

// Build SDPA-based attention (primary fast path)
ov::Output<ov::Node> build_sdpa(const ov::Output<ov::Node>& Q,
                                const ov::Output<ov::Node>& K,
                                const ov::Output<ov::Node>& V,
                                bool has_mask,
                                const ov::Output<ov::Node>& attn_mask,
                                float scale_attr,
                                bool is_causal) {
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
    return std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal)->output(0);
}

// Build manual attention decomposition (for softcap or qk_matmul_output)
// Returns {Y, qk_matmul_output_or_null}
ov::OutputVector build_manual_attention(const ov::Output<ov::Node>& Q,
                                        const ov::Output<ov::Node>& K,
                                        const ov::Output<ov::Node>& V,
                                        bool has_mask,
                                        const ov::Output<ov::Node>& attn_mask,
                                        float scale_attr,
                                        float softcap,
                                        bool is_causal,
                                        int64_t qk_matmul_output_mode,
                                        bool needs_qk_output) {
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

    // 3. Apply attention mask and causal mask
    std::shared_ptr<ov::Node> masked = scaled_qk;
    if (has_mask) {
        masked = std::make_shared<v1::Add>(scaled_qk, attn_mask);
    }
    if (is_causal) {
        auto causal_mask = build_causal_mask(Q, K);
        masked = std::make_shared<v1::Add>(masked, causal_mask);
    }

    // Capture qk_matmul_output at mode 0 (raw QK) or mode 1 (after mask)
    ov::Output<ov::Node> qk_debug_output;
    if (needs_qk_output && qk_matmul_output_mode == 0) {
        qk_debug_output = scaled_qk->output(0);
    } else if (needs_qk_output && qk_matmul_output_mode == 1) {
        qk_debug_output = masked->output(0);
    }

    // 4. Apply softcap: softcap * tanh(scores / softcap)
    std::shared_ptr<ov::Node> capped = masked;
    if (softcap > 0.0f) {
        auto cap = v0::Constant::create(Q.get_element_type(), ov::Shape{}, {softcap});
        auto divided = std::make_shared<v1::Divide>(masked, cap);
        auto tanh_out = std::make_shared<v0::Tanh>(divided);
        capped = std::make_shared<v1::Multiply>(tanh_out, cap);
    }

    // Capture at mode 2 (after softcap)
    if (needs_qk_output && qk_matmul_output_mode == 2) {
        qk_debug_output = capped->output(0);
    }

    // 5. Softmax
    auto softmax_out = std::make_shared<v8::Softmax>(capped, -1);

    // Capture at mode 3 (after softmax)
    if (needs_qk_output && qk_matmul_output_mode == 3) {
        qk_debug_output = softmax_out->output(0);
    }

    // 6. softmax @ V
    auto output = std::make_shared<v0::MatMul>(softmax_out, V);

    ov::OutputVector results;
    results.push_back(output->output(0));
    if (needs_qk_output && qk_debug_output.get_node()) {
        results.push_back(qk_debug_output);
    } else {
        results.push_back(ov::Output<ov::Node>{});
    }
    return results;
}

}  // namespace
}  // namespace detail

namespace opset_23 {
ov::OutputVector attention(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();
    CHECK_VALID_NODE(node, num_inputs >= 3, "Attention expects at least 3 inputs, got: ", num_inputs);

    // Required inputs
    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];

    // Optional inputs
    bool has_attn_mask = num_inputs > 3 && !ov::op::util::is_null(inputs[3]);
    bool has_past_key = num_inputs > 4 && !ov::op::util::is_null(inputs[4]);
    bool has_past_value = num_inputs > 5 && !ov::op::util::is_null(inputs[5]);

    CHECK_VALID_NODE(node,
                     has_past_key == has_past_value,
                     "past_key and past_value must be both present or both absent");

    // Attributes
    bool is_causal = static_cast<bool>(node.get_attribute_value<int64_t>("is_causal", 0));
    float scale_attr = node.get_attribute_value<float>("scale", 0.0f);
    float softcap = node.get_attribute_value<float>("softcap", 0.0f);
    int64_t qk_matmul_output_mode = node.get_attribute_value<int64_t>("qk_matmul_output_mode", 0);
    int64_t q_num_heads = node.get_attribute_value<int64_t>("q_num_heads", 0);
    int64_t kv_num_heads = node.get_attribute_value<int64_t>("kv_num_heads", 0);

    CHECK_VALID_NODE(node, softcap >= 0.0f, "softcap must be non-negative, got: ", softcap);

    // Determine number of requested outputs
    size_t num_outputs = node.get_outputs_size();
    bool needs_qk_output = num_outputs > 3;

    // Determine if inputs are 3D and need reshaping
    auto q_rank = Q.get_partial_shape().rank();
    auto k_rank = K.get_partial_shape().rank();
    bool q_is_3d = false;
    bool kv_is_3d = false;

    if (q_rank.is_static()) {
        q_is_3d = (q_rank.get_length() == 3);
        CHECK_VALID_NODE(node,
                         q_rank.get_length() == 3 || q_rank.get_length() == 4,
                         "Q input rank must be 3 or 4, got: ",
                         q_rank.get_length());
    }
    if (k_rank.is_static()) {
        kv_is_3d = (k_rank.get_length() == 3);
        CHECK_VALID_NODE(node,
                         k_rank.get_length() == 3 || k_rank.get_length() == 4,
                         "K input rank must be 3 or 4, got: ",
                         k_rank.get_length());
    }

    // Reshape 3D inputs to 4D
    if (q_is_3d) {
        CHECK_VALID_NODE(node, q_num_heads > 0, "q_num_heads attribute is required for 3D Q input");
        Q = detail::reshape_3d_to_4d(Q, q_num_heads);
    }
    if (kv_is_3d) {
        CHECK_VALID_NODE(node, kv_num_heads > 0, "kv_num_heads attribute is required for 3D K/V inputs");
        K = detail::reshape_3d_to_4d(K, kv_num_heads);
        V = detail::reshape_3d_to_4d(V, kv_num_heads);
    }

    // Handle KV cache: concatenate past K/V with current K/V along sequence dim (axis=2)
    if (has_past_key) {
        K = std::make_shared<v0::Concat>(ov::OutputVector{inputs[4], K}, 2);
        V = std::make_shared<v0::Concat>(ov::OutputVector{inputs[5], V}, 2);
    }

    // present_key and present_value are the K and V after concatenation (before head expansion)
    auto present_key = K;
    auto present_value = V;

    // Handle GQA: expand K/V heads to match Q heads
    // Case 1: head counts from explicit attributes (3D inputs)
    if (q_num_heads > 0 && kv_num_heads > 0 && q_num_heads != kv_num_heads) {
        CHECK_VALID_NODE(node,
                         q_num_heads % kv_num_heads == 0,
                         "q_num_heads must be divisible by kv_num_heads for GQA. q_num_heads=",
                         q_num_heads,
                         ", kv_num_heads=",
                         kv_num_heads);
        int64_t n_rep = q_num_heads / kv_num_heads;
        K = detail::repeat_kv(K, n_rep);
        V = detail::repeat_kv(V, n_rep);
    }
    // Case 2: head counts from shapes (4D inputs without attributes)
    else {
        auto q_pshape = Q.get_partial_shape();
        auto k_pshape = K.get_partial_shape();
        if (q_pshape.rank().is_static() && q_pshape.rank().get_length() == 4 &&
            k_pshape.rank().is_static() && k_pshape.rank().get_length() == 4 &&
            q_pshape[1].is_static() && k_pshape[1].is_static()) {
            auto q_heads = q_pshape[1].get_length();
            auto kv_heads = k_pshape[1].get_length();
            if (q_heads != kv_heads) {
                CHECK_VALID_NODE(node,
                                 q_heads % kv_heads == 0,
                                 "q_heads must be divisible by kv_heads for GQA. q_heads=",
                                 q_heads,
                                 ", kv_heads=",
                                 kv_heads);
                int64_t n_rep = static_cast<int64_t>(q_heads / kv_heads);
                K = detail::repeat_kv(K, n_rep);
                V = detail::repeat_kv(V, n_rep);
            }
        }
    }

    // Prepare attention mask
    ov::Output<ov::Node> attn_mask;
    if (has_attn_mask) {
        attn_mask = inputs[3];
        if (attn_mask.get_element_type() == ov::element::boolean) {
            // For manual path, convert boolean to float; for SDPA path, SDPA handles boolean natively
            if (softcap > 0.0f || needs_qk_output) {
                attn_mask = detail::convert_boolean_mask(attn_mask, Q.get_element_type());
            }
        }
    }

    // When KV cache is used with is_causal, SDPA's internal causal mask doesn't account for
    // KV cache offset (seq_kv > seq_q). Build an explicit offset-aware causal mask and pass it
    // as attn_mask instead, disabling SDPA's is_causal flag.
    if (is_causal && has_past_key) {
        auto causal_mask = detail::build_causal_mask(Q, K);
        if (has_attn_mask) {
            attn_mask = std::make_shared<v1::Add>(attn_mask, causal_mask);
        } else {
            attn_mask = causal_mask;
        }
        has_attn_mask = true;
        is_causal = false;
    }

    // Choose execution path
    ov::Output<ov::Node> Y;
    ov::Output<ov::Node> qk_debug_output;

    if (softcap > 0.0f || needs_qk_output) {
        // Manual decomposition path (softcap or debug output)
        auto results = detail::build_manual_attention(Q,
                                                      K,
                                                      V,
                                                      has_attn_mask,
                                                      attn_mask,
                                                      scale_attr,
                                                      softcap,
                                                      is_causal,
                                                      qk_matmul_output_mode,
                                                      needs_qk_output);
        Y = results[0];
        if (needs_qk_output && results.size() > 1 && results[1].get_node()) {
            qk_debug_output = results[1];
        }
    } else {
        // SDPA path (primary fast path)
        Y = detail::build_sdpa(Q, K, V, has_attn_mask, attn_mask, scale_attr, is_causal);
    }

    // Reshape output back to 3D if Q was 3D
    if (q_is_3d) {
        Y = detail::reshape_4d_to_3d(Y);
    }

    // Build output vector.
    // Output names from the ONNX graph determine which outputs are actually requested.
    // Empty names indicate unused optional outputs — push NullNode for those to avoid
    // creating shared input/output parameters that confuse port resolution.
    auto output_names = node.get_output_names();
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
ONNX_OP("Attention", OPSET_SINCE(23), ai_onnx::opset_23::attention);
}  // namespace opset_23

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
