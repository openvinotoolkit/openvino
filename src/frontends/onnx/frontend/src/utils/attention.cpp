// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/attention.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace attention {

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<v3::ShapeOf>& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims) {
    return get_dimensions(std::make_shared<v3::ShapeOf>(node), dims);
}

// Convert boolean mask to float additive mask: true -> 0.0, false -> -10000.0
ov::Output<ov::Node> convert_boolean_mask(const ov::Output<ov::Node>& mask,
                                          const ov::element::Type& type,
                                          float mask_filter_value) {
    auto zero = v0::Constant::create(type, ov::Shape{}, {0.0f});
    auto neg_large = v0::Constant::create(type, ov::Shape{}, {mask_filter_value});
    return std::make_shared<v1::Select>(mask, zero, neg_large);
}

// Build additive causal mask of shape (seq_q, seq_kv): 0 for allowed, -10000 for masked.
// When use_offset=true, accounts for KV cache offset so that query position i attends to
// key positions j where j <= i + (seq_kv - seq_q). Use this for KV cache scenarios.
// When use_offset=false, builds a simple lower-triangular mask matching np.tril(k=0),
// where query position i attends to key positions j where j <= i.
ov::Output<ov::Node> build_causal_mask(const ov::Output<ov::Node>& Q, const ov::Output<ov::Node>& K, bool use_offset) {
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
        auto causal_mask = build_causal_mask(Q, K, false);
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

}  // namespace attention
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
