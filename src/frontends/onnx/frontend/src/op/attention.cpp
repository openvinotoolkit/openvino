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
#include "openvino/op/divide.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using namespace ov::frontend::onnx::attention;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace detail {
namespace {

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
    bool has_attn_mask = common::is_input_valid(node, 3);
    bool has_past_key = common::is_input_valid(node, 4);
    bool has_past_value = common::is_input_valid(node, 5);

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
    CHECK_VALID_NODE(node,
                     qk_matmul_output_mode >= 0 && qk_matmul_output_mode <= 3,
                     "qk_matmul_output_mode must be 0, 1, 2, or 3, got: ",
                     qk_matmul_output_mode);

    // Determine number of requested outputs
    size_t num_outputs = node.get_outputs_size();
    const auto& output_names = node.get_output_names();
    bool needs_qk_output = output_names.size() > 3 && !output_names[3].get().empty();

    // Determine if inputs are 3D and need reshaping
    auto q_rank = Q.get_partial_shape().rank();
    auto k_rank = K.get_partial_shape().rank();
    auto v_rank = V.get_partial_shape().rank();
    bool q_is_3d = false;
    bool kv_is_3d = false;

    if (q_rank.is_static()) {
        q_is_3d = (q_rank.get_length() == 3);
        CHECK_VALID_NODE(node,
                         q_is_3d || q_rank.get_length() == 4,
                         "Q input rank must be 3 or 4, got: ",
                         q_rank.get_length());
    }
    if (k_rank.is_static()) {
        kv_is_3d = (k_rank.get_length() == 3);
        CHECK_VALID_NODE(node,
                         kv_is_3d || k_rank.get_length() == 4,
                         "K input rank must be 3 or 4, got: ",
                         k_rank.get_length());
    }
    if (v_rank.is_static()) {
        CHECK_VALID_NODE(node,
                         v_rank.get_length() == 3 || v_rank.get_length() == 4,
                         "V input rank must be 3 or 4, got: ",
                         v_rank.get_length());
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
        if (q_pshape.rank().is_static() && q_pshape.rank().get_length() == 4 && k_pshape.rank().is_static() &&
            k_pshape.rank().get_length() == 4 && q_pshape[1].is_static() && k_pshape[1].is_static()) {
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
                attn_mask = convert_boolean_mask(attn_mask, Q.get_element_type());
            }
        }
    }

    // Build an explicit causal mask instead of relying on SDPA's internal is_causal flag.
    // SDPA's internal mask uses offset-based semantics (ncausal = kv_len - q_len + m + 1)
    // which doesn't match the ONNX spec's np.tril(k=0) for non-square attention matrices
    // (seq_q != seq_kv). For KV cache scenarios, use offset to account for past sequence.
    if (is_causal) {
        auto causal_mask = build_causal_mask(Q, K, has_past_key);
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
        auto results = build_manual_attention(Q,
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
        Y = build_sdpa(Q, K, V, has_attn_mask, attn_mask, scale_attr, is_causal);
    }

    // Reshape output back to 3D if Q was 3D
    if (q_is_3d) {
        Y = detail::reshape_4d_to_3d(Y);
    }

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
ONNX_OP("Attention", OPSET_SINCE(1), ai_onnx::opset_23::attention);
}  // namespace opset_23

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
