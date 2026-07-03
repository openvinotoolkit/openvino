// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/attention.hpp"

#include <limits>

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using namespace ov::frontend::onnx::attention;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace detail {
namespace {

// Repeat K/V heads for GQA: (B, kv_h, seq, h) -> (B, kv_h * n_rep, seq, h).
// Interleave-repeats each KV head n_rep times ([h0]*n_rep, [h1]*n_rep, ...) to match the ONNX
// reference's np.repeat(K, n_rep, axis=1): contiguous query-head groups share one KV head.
// n_rep = q_num_heads / kv_num_heads
ov::Output<ov::Node> repeat_kv(const ov::Output<ov::Node>& input, int64_t n_rep) {
    if (n_rep == 1) {
        return input;
    }
    auto shape = std::make_shared<v3::ShapeOf>(input);
    auto batch = get_dimensions(shape, {0});
    auto kv_h = get_dimensions(shape, {1});
    auto seq = get_dimensions(shape, {2});
    auto head = get_dimensions(shape, {3});
    auto n_rep_node = v0::Constant::create(ov::element::i64, ov::Shape{1}, {n_rep});
    auto new_heads = std::make_shared<v1::Multiply>(kv_h, n_rep_node);  // kv_h * n_rep

    // (B, kv_h, seq, h) -> (B, kv_h, 1, seq, h) -> (B, kv_h, n_rep, seq, h)
    auto axis2 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    auto unsq = std::make_shared<v0::Unsqueeze>(input, axis2);
    auto tile_repeats = v0::Constant::create(ov::element::i64, ov::Shape{5}, std::vector<int64_t>{1, 1, n_rep, 1, 1});
    auto tiled = std::make_shared<v0::Tile>(unsq, tile_repeats);

    // merge (kv_h, n_rep) -> kv_h*n_rep : (B, kv_h*n_rep, seq, h)
    auto new_shape = std::make_shared<v0::Concat>(ov::OutputVector{batch, new_heads, seq, head}, 0);
    return std::make_shared<v1::Reshape>(tiled, new_shape, false);
}

const float NEG_INF = -std::numeric_limits<float>::infinity();

// Build an additive padding mask (-inf for disallowed) from nonpad_kv_seqlen.
// Key position j is valid for batch b iff j < nonpad[b]. Result shape: (B, 1, 1, seq_kv).
ov::Output<ov::Node> build_padding_mask(const ov::Output<ov::Node>& K, const ov::Output<ov::Node>& nonpad) {
    auto k_shape = std::make_shared<v3::ShapeOf>(K);
    auto seq_kv = get_dimensions(k_shape, {2});  // (1,)
    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto seq_kv_s = std::make_shared<v0::Squeeze>(seq_kv, zero);
    auto col = std::make_shared<v4::Range>(zero, seq_kv_s, one, ov::element::i64);  // (seq_kv,)

    auto col_shape = v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, 1, -1});
    auto col_4d = std::make_shared<v1::Reshape>(col, col_shape, false);  // (1, 1, 1, seq_kv)
    auto nonpad_shape = v0::Constant::create(ov::element::i64, ov::Shape{4}, {-1, 1, 1, 1});
    auto nonpad_4d = std::make_shared<v1::Reshape>(nonpad, nonpad_shape, false);  // (B, 1, 1, 1)

    auto allowed = std::make_shared<v1::Less>(col_4d, nonpad_4d);  // (B, 1, 1, seq_kv)
    return convert_boolean_mask(allowed, K.get_element_type());
}

// Pad the last dimension of an additive float mask up to total_kv with -inf (opset-24 allows the
// attention mask's last dimension to be shorter than the total sequence length). Requires a static
// mask rank; returns the mask unchanged when the rank is dynamic.
ov::Output<ov::Node> pad_attn_mask_last_dim(const ov::Output<ov::Node>& mask, const ov::Output<ov::Node>& K) {
    const auto rank = mask.get_partial_shape().rank();
    if (rank.is_dynamic()) {
        return mask;
    }
    const auto r = static_cast<size_t>(rank.get_length());
    const auto& compute_type = K.get_element_type();

    auto mask_shape = std::make_shared<v3::ShapeOf>(mask);
    auto k_shape = std::make_shared<v3::ShapeOf>(K);
    auto cur_last = get_dimensions(mask_shape, {static_cast<int>(r) - 1});  // (1,)
    auto total_kv = get_dimensions(k_shape, {2});                           // (1,)

    auto pad_amount = std::make_shared<v1::Subtract>(total_kv, cur_last);  // (1,)

    auto pads_begin = v0::Constant::create(ov::element::i64, ov::Shape{r}, std::vector<int64_t>(r, 0));
    auto zeros_head = v0::Constant::create(ov::element::i64, ov::Shape{r - 1}, std::vector<int64_t>(r - 1, 0));
    auto pads_end = std::make_shared<v0::Concat>(ov::OutputVector{zeros_head, pad_amount}, 0);  // (r,)

    auto pad_value = v0::Constant::create(compute_type, ov::Shape{}, {NEG_INF});
    return std::make_shared<v12::Pad>(mask, pads_begin, pads_end, pad_value, ov::op::PadMode::CONSTANT);
}

// Result of the shared Attention prologue.
struct PreparedQKV {
    ov::Output<ov::Node> Q;
    ov::Output<ov::Node> K;
    ov::Output<ov::Node> V;
    ov::Output<ov::Node> present_key;
    ov::Output<ov::Node> present_value;
    bool q_is_3d = false;
};

// Shared prologue for ONNX Attention opset-23/-24: validates input ranks, reshapes 3D inputs to 4D,
// concatenates the KV cache along the sequence axis, captures present_key/present_value (before head
// expansion) and expands K/V heads for grouped-query attention. This logic is identical across the
// two opset versions; only the mask construction and attention core differ.
PreparedQKV prepare_qkv(const ov::frontend::onnx::Node& node,
                        const ov::OutputVector& inputs,
                        bool has_past_key,
                        int64_t q_num_heads,
                        int64_t kv_num_heads) {
    PreparedQKV out;
    out.Q = inputs[0];
    out.K = inputs[1];
    out.V = inputs[2];

    auto q_rank = out.Q.get_partial_shape().rank();
    auto k_rank = out.K.get_partial_shape().rank();
    auto v_rank = out.V.get_partial_shape().rank();
    bool kv_is_3d = false;

    if (q_rank.is_static()) {
        out.q_is_3d = (q_rank.get_length() == 3);
        CHECK_VALID_NODE(node,
                         out.q_is_3d || q_rank.get_length() == 4,
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

    if (out.q_is_3d) {
        CHECK_VALID_NODE(node, q_num_heads > 0, "q_num_heads attribute is required for 3D Q input");
        out.Q = reshape_3d_to_4d(out.Q, q_num_heads);
    }
    if (kv_is_3d) {
        CHECK_VALID_NODE(node, kv_num_heads > 0, "kv_num_heads attribute is required for 3D K/V inputs");
        out.K = reshape_3d_to_4d(out.K, kv_num_heads);
        out.V = reshape_3d_to_4d(out.V, kv_num_heads);
    }

    // KV cache: concatenate past K/V with current K/V along the sequence dim (axis=2).
    if (has_past_key) {
        out.K = std::make_shared<v0::Concat>(ov::OutputVector{inputs[4], out.K}, 2);
        out.V = std::make_shared<v0::Concat>(ov::OutputVector{inputs[5], out.V}, 2);
    }

    // present_key/present_value are K/V after concatenation, before head expansion.
    out.present_key = out.K;
    out.present_value = out.V;

    // GQA head expansion.
    if (q_num_heads > 0 && kv_num_heads > 0 && q_num_heads != kv_num_heads) {
        CHECK_VALID_NODE(node,
                         q_num_heads % kv_num_heads == 0,
                         "q_num_heads must be divisible by kv_num_heads for GQA. q_num_heads=",
                         q_num_heads,
                         ", kv_num_heads=",
                         kv_num_heads);
        int64_t n_rep = q_num_heads / kv_num_heads;
        out.K = repeat_kv(out.K, n_rep);
        out.V = repeat_kv(out.V, n_rep);
    } else {
        auto q_pshape = out.Q.get_partial_shape();
        auto k_pshape = out.K.get_partial_shape();
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
                out.K = repeat_kv(out.K, n_rep);
                out.V = repeat_kv(out.V, n_rep);
            }
        }
    }

    return out;
}

}  // namespace
}  // namespace detail

namespace opset_23 {
ov::OutputVector attention(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();
    CHECK_VALID_NODE(node, num_inputs >= 3, "Attention expects at least 3 inputs, got: ", num_inputs);

    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];

    bool has_attn_mask = common::is_input_valid(node, 3);
    bool has_past_key = common::is_input_valid(node, 4);
    bool has_past_value = common::is_input_valid(node, 5);

    CHECK_VALID_NODE(node,
                     has_past_key == has_past_value,
                     "past_key and past_value must be both present or both absent");

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

    size_t num_outputs = node.get_outputs_size();
    const auto& output_names = node.get_output_names();
    if (num_outputs > 1) {
        bool has_present_key = num_outputs > 1 && !output_names[1].get().empty();
        bool has_present_value = num_outputs > 2 && !output_names[2].get().empty();
        CHECK_VALID_NODE(node,
                         has_present_key == has_present_value,
                         "present_key and present_value must be both present or both absent");
    }

    bool needs_qk_output = output_names.size() > 3 && !output_names[3].get().empty();
    if (!needs_qk_output)
        qk_matmul_output_mode = -1;

    auto prepared = detail::prepare_qkv(node, inputs, has_past_key, q_num_heads, kv_num_heads);
    Q = prepared.Q;
    K = prepared.K;
    V = prepared.V;
    auto present_key = prepared.present_key;
    auto present_value = prepared.present_value;
    bool q_is_3d = prepared.q_is_3d;

    const auto& compute_type = Q.get_element_type();

    ov::Output<ov::Node> attn_mask;
    if (has_attn_mask) {
        attn_mask = inputs[3];
        CHECK_VALID_NODE(
            node,
            attn_mask.get_element_type() == ov::element::boolean || attn_mask.get_element_type() == compute_type,
            "Attention mask must be boolean or match Q/K/V type (",
            compute_type,
            "), got: ",
            attn_mask.get_element_type());
        if (attn_mask.get_element_type() == ov::element::boolean) {
            // For manual path, convert boolean to float; for SDPA path, SDPA op. handles boolean natively.
            // The default -inf fill lets build_manual_attention's fully-masked-row guard detect a row
            // whose keys are all disallowed and emit a zero row (ONNX Attention-23 semantics).
            if (softcap > 0.0f || qk_matmul_output_mode >= 0 || is_causal) {
                attn_mask = convert_boolean_mask(attn_mask, compute_type);
            }
        }
    } else {
        attn_mask = std::make_shared<v0::Constant>(compute_type, ov::Shape{}, 0.0f);
    }

    // Build an explicit causal mask rather than using the SDPA op's is_causal flag, because the flag
    // is implemented inconsistently across backends (CPU: bottom-right offset, GPU: top-left).
    // An explicit additive mask gives identical results on every backend. The mask uses
    // offset = past_sequence_length when a KV cache is present and 0 otherwise — the bottom-right
    // alignment according to the ONNX Attention spec (confirmed by onnx/onnx#8068).
    if (is_causal) {
        auto kind = has_past_key ? CausalKind::PAST : CausalKind::NONE;
        auto causal_mask = build_causal_mask(Q, K, kind);
        if (has_attn_mask) {
            attn_mask = std::make_shared<v1::Add>(attn_mask, causal_mask);
        } else {
            attn_mask = causal_mask;
        }
        has_attn_mask = true;
        is_causal = false;
    }

    ov::Output<ov::Node> Y;
    ov::Output<ov::Node> qk_debug_output;

    if (softcap > 0.0f || qk_matmul_output_mode >= 0) {
        auto results = build_manual_attention(Q, K, V, attn_mask, scale_attr, softcap, qk_matmul_output_mode);
        Y = results[0];
        if (results[1].get_node()) {
            qk_debug_output = results[1];
        }
    } else {
        ov::OutputVector inputs{Q, K, V, attn_mask};
        if (scale_attr != 0.0f)
            inputs.push_back(v0::Constant::create(compute_type, ov::Shape{}, {scale_attr}));

        Y = std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal)->output(0);
    }

    // Reshape output back to 3D if Q was 3D
    if (q_is_3d)
        Y = reshape_4d_to_3d(Y);

    // Output names from the ONNX graph determine which outputs are actually requested.
    // Empty names indicate unused optional outputs — push NullNode for those to avoid
    // creating shared input/output parameters that confuse port resolution.
    ov::OutputVector results{Y};

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
ONNX_OP("Attention", OPSET_RANGE(1, 23), ai_onnx::opset_23::attention);
}  // namespace opset_23

namespace opset_24 {
ov::OutputVector attention(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    const auto num_inputs = inputs.size();
    CHECK_VALID_NODE(node, num_inputs >= 3, "Attention expects at least 3 inputs, got: ", num_inputs);

    auto Q = inputs[0];
    auto K = inputs[1];
    auto V = inputs[2];

    bool has_attn_mask = common::is_input_valid(node, 3);
    bool has_past_key = common::is_input_valid(node, 4);
    bool has_past_value = common::is_input_valid(node, 5);
    bool has_nonpad = common::is_input_valid(node, 6);

    CHECK_VALID_NODE(node,
                     has_past_key == has_past_value,
                     "past_key and past_value must be both present or both absent");
    CHECK_VALID_NODE(node,
                     !(has_nonpad && (has_past_key || has_past_value)),
                     "nonpad_kv_seqlen is mutually exclusive with past_key/past_value");

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

    size_t num_outputs = node.get_outputs_size();
    const auto& output_names = node.get_output_names();
    if (num_outputs > 1) {
        bool has_present_key = num_outputs > 1 && !output_names[1].get().empty();
        bool has_present_value = num_outputs > 2 && !output_names[2].get().empty();
        CHECK_VALID_NODE(node,
                         has_present_key == has_present_value,
                         "present_key and present_value must be both present or both absent");
    }
    bool needs_qk_output = output_names.size() > 3 && !output_names[3].get().empty();
    if (!needs_qk_output)
        qk_matmul_output_mode = -1;

    auto prepared = detail::prepare_qkv(node, inputs, has_past_key, q_num_heads, kv_num_heads);
    Q = prepared.Q;
    K = prepared.K;
    V = prepared.V;
    auto present_key = prepared.present_key;
    auto present_value = prepared.present_value;
    bool q_is_3d = prepared.q_is_3d;

    const auto& compute_type = Q.get_element_type();

    // Merge attention mask, causal mask and padding mask into a single additive mask using -inf for
    // disallowed positions (required for the fully-masked-row guard in build_manual_attention).
    ov::Output<ov::Node> attn_mask;
    if (has_attn_mask) {
        attn_mask = inputs[3];
        CHECK_VALID_NODE(
            node,
            attn_mask.get_element_type() == ov::element::boolean || attn_mask.get_element_type() == compute_type,
            "Attention mask must be boolean or match Q/K/V type (",
            compute_type,
            "), got: ",
            attn_mask.get_element_type());
        if (attn_mask.get_element_type() == ov::element::boolean) {
            attn_mask = convert_boolean_mask(attn_mask, compute_type);
        }
        // The mask's last dimension may be shorter than the total sequence length; pad with -inf.
        attn_mask = detail::pad_attn_mask_last_dim(attn_mask, K);
    } else {
        attn_mask = std::make_shared<v0::Constant>(compute_type, ov::Shape{}, 0.0f);
    }

    ov::Output<ov::Node> nonpad;
    if (has_nonpad) {
        nonpad = inputs[6];
    }

    if (is_causal) {
        CausalKind kind = has_past_key ? CausalKind::PAST : has_nonpad ? CausalKind::NONPAD : CausalKind::NONE;
        auto causal_mask = build_causal_mask(Q, K, kind, detail::NEG_INF, nonpad);
        if (has_attn_mask) {
            attn_mask = std::make_shared<v1::Add>(attn_mask, causal_mask);
        } else {
            attn_mask = causal_mask;
        }
        has_attn_mask = true;
    }

    if (has_nonpad) {
        auto padding_mask = detail::build_padding_mask(K, nonpad);
        if (has_attn_mask) {
            attn_mask = std::make_shared<v1::Add>(attn_mask, padding_mask);
        } else {
            attn_mask = padding_mask;
        }
        has_attn_mask = true;
    }

    ov::Output<ov::Node> Y;
    ov::Output<ov::Node> qk_debug_output;

    if (softcap > 0.0f || qk_matmul_output_mode >= 0 || has_nonpad) {
        auto results = build_manual_attention(Q, K, V, attn_mask, scale_attr, softcap, qk_matmul_output_mode);
        Y = results[0];
        if (results[1].get_node()) {
            qk_debug_output = results[1];
        }
    } else {
        ov::OutputVector inputs{Q, K, V, attn_mask};
        if (scale_attr != 0.0f)
            inputs.push_back(v0::Constant::create(compute_type, ov::Shape{}, {scale_attr}));

        Y = std::make_shared<v13::ScaledDotProductAttention>(inputs, is_causal)->output(0);
    }

    if (q_is_3d) {
        Y = reshape_4d_to_3d(Y);
    }

    ov::OutputVector results{Y};

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
ONNX_OP("Attention", OPSET_SINCE(24), ai_onnx::opset_24::attention);
}  // namespace opset_24

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
