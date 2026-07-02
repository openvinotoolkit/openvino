// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/attention.hpp"

#include <limits>

#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/less_eq.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
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

// Reshape 3D input (batch, seq, num_heads * head_size) to 4D (batch, num_heads, seq, head_size).
ov::Output<ov::Node> reshape_3d_to_4d(const ov::Output<ov::Node>& input, int64_t num_heads) {
    auto reshape_pattern =
        v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 0, num_heads, -1});
    auto reshaped = std::make_shared<v1::Reshape>(input, reshape_pattern, true);
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    return std::make_shared<v1::Transpose>(reshaped, perm);
}

// Reshape 4D output (batch, num_heads, seq, head_size) back to 3D (batch, seq, num_heads * head_size).
ov::Output<ov::Node> reshape_4d_to_3d(const ov::Output<ov::Node>& output) {
    auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});
    auto transposed = std::make_shared<v1::Transpose>(output, perm);
    auto reshape_pattern = v0::Constant::create(ov::element::i64, ov::Shape{3}, {0, 0, -1});
    return std::make_shared<v1::Reshape>(transposed, reshape_pattern, true);
}

// Convert boolean mask to float additive mask: true -> 0.0, false -> mask_filter_value (default -inf).
ov::Output<ov::Node> convert_boolean_mask(const ov::Output<ov::Node>& mask,
                                          const ov::element::Type& type,
                                          float mask_filter_value) {
    auto zero = v0::Constant::create(type, ov::Shape{}, {0.0f});
    auto neg_large = v0::Constant::create(type, ov::Shape{}, {mask_filter_value});
    return std::make_shared<v1::Select>(mask, zero, neg_large);
}

// Build an additive causal mask with opset-24 bottom-right alignment. For CausalKind::NONPAD the offset
// varies per batch and the result has shape (B, 1, seq_q, seq_kv); otherwise the result is (seq_q, seq_kv)
// and broadcasts over batch and heads.
ov::Output<ov::Node> build_causal_mask(const ov::Output<ov::Node>& Q,
                                       const ov::Output<ov::Node>& K,
                                       CausalKind kind,
                                       float mask_filter_value,
                                       const ov::Output<ov::Node>& nonpad) {
    auto q_shape = std::make_shared<v3::ShapeOf>(Q);
    auto k_shape = std::make_shared<v3::ShapeOf>(K);
    auto seq_q = get_dimensions(q_shape, {2});   // (1,)
    auto seq_kv = get_dimensions(k_shape, {2});  // (1,)

    auto zero = v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
    auto one = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});

    auto seq_q_scalar = std::make_shared<v0::Squeeze>(seq_q, zero);
    auto seq_kv_scalar = std::make_shared<v0::Squeeze>(seq_kv, zero);

    auto col = std::make_shared<v4::Range>(zero, seq_kv_scalar, one, ov::element::i64);  // (seq_kv,)
    auto row = std::make_shared<v4::Range>(zero, seq_q_scalar, one, ov::element::i64);   // (seq_q,)

    // diff[i, j] = col[j] - row[i]  -> (seq_q, seq_kv)
    auto axis1 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto row_2d = std::make_shared<v0::Unsqueeze>(row, axis1);  // (seq_q, 1)
    auto diff = std::make_shared<v1::Subtract>(col, row_2d);    // (seq_q, seq_kv)

    std::shared_ptr<ov::Node> allowed;
    if (kind == CausalKind::NONPAD) {
        // offset_b = nonpad[b] - seq_q ; allowed[b, i, j] = diff[i, j] <= offset_b
        auto offset = std::make_shared<v1::Subtract>(nonpad, seq_q);  // (B,)
        auto offset_shape = v0::Constant::create(ov::element::i64, ov::Shape{3}, {-1, 1, 1});
        auto offset_3d = std::make_shared<v1::Reshape>(offset, offset_shape, false);  // (B, 1, 1)
        auto axis0 = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto diff_3d = std::make_shared<v0::Unsqueeze>(diff, axis0);            // (1, seq_q, seq_kv)
        auto allowed_3d = std::make_shared<v1::LessEqual>(diff_3d, offset_3d);  // (B, seq_q, seq_kv)
        allowed = std::make_shared<v0::Unsqueeze>(allowed_3d, axis1);           // (B, 1, seq_q, seq_kv)
    } else if (kind == CausalKind::PAST) {
        auto offset = std::make_shared<v1::Subtract>(seq_kv_scalar, seq_q_scalar);  // scalar
        allowed = std::make_shared<v1::LessEqual>(diff, offset);
    } else {
        allowed = std::make_shared<v1::LessEqual>(diff, zero);
    }

    return convert_boolean_mask(allowed, Q.get_element_type(), mask_filter_value);
}

// Build manual attention decomposition following ONNX opset-24 semantics.
// The mask passed in is already merged (attention + causal + padding) using -inf for disallowed
// positions. Order of operations: scale -> softcap -> mask -> softmax. When include_safe_softmax is
// true the fully-masked-row guard is applied (opset-23/-24); MultiHeadAttention passes false.
ov::OutputVector build_manual_attention(const ov::Output<ov::Node>& Q,
                                        const ov::Output<ov::Node>& K,
                                        const ov::Output<ov::Node>& V,
                                        const ov::Output<ov::Node>& attn_mask,
                                        float scale_attr,
                                        float softcap,
                                        int64_t qk_matmul_output_mode,
                                        bool include_safe_softmax) {
    const auto& compute_type = Q.get_element_type();

    // 1. Q @ K^T
    auto qk = std::make_shared<v0::MatMul>(Q, K, false, true);

    // 2. Apply scale
    std::shared_ptr<ov::Node> scaled_qk;
    if (scale_attr != 0.0f) {
        auto scale_node = v0::Constant::create(compute_type, ov::Shape{}, {scale_attr});
        scaled_qk = std::make_shared<v1::Multiply>(qk, scale_node);
    } else {
        auto q_shape = std::make_shared<v3::ShapeOf>(Q);
        auto head_size = get_dimensions(q_shape, {3});
        auto head_size_f = std::make_shared<v0::Convert>(head_size, compute_type);
        auto sqrt_head = std::make_shared<v0::Sqrt>(head_size_f);
        scaled_qk = std::make_shared<v1::Divide>(qk, sqrt_head);
    }

    // 3. Apply softcap: softcap * tanh(scores / softcap)
    std::shared_ptr<ov::Node> capped = scaled_qk;
    if (softcap > 0.0f) {
        auto cap = v0::Constant::create(compute_type, ov::Shape{}, {softcap});
        auto divided = std::make_shared<v1::Divide>(scaled_qk, cap);
        auto tanh_out = std::make_shared<v0::Tanh>(divided);
        capped = std::make_shared<v1::Multiply>(tanh_out, cap);
    }

    // 4. Apply the (already merged) additive attention mask
    std::shared_ptr<ov::Node> masked = std::make_shared<v1::Add>(capped, attn_mask);

    // 5. Softmax with an optional fully-masked-row guard: rows whose keys are all masked out (additive
    // mask of -inf) would yield NaN after softmax; opset-24 requires a zero output row instead.
    std::shared_ptr<ov::Node> softmax_out = std::make_shared<v8::Softmax>(masked, -1);

    if (include_safe_softmax) {
        bool non_empty_attn_mask = true;
        if (auto mask_const = ov::as_type_ptr<v0::Constant>(attn_mask.get_node_shared_ptr())) {
            if (ov::shape_size(mask_const->get_shape()) == 1 && mask_const->cast_vector<float>()[0] == 0.0f)
                non_empty_attn_mask = false;
        }

        if (non_empty_attn_mask) {
            auto finite_threshold =
                v0::Constant::create(compute_type, ov::Shape{}, {std::numeric_limits<float>::lowest()});
            auto reduce_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
            auto row_max = std::make_shared<v1::ReduceMax>(masked, reduce_axis, true);
            auto row_valid = std::make_shared<v1::Greater>(row_max, finite_threshold);
            auto zero = v0::Constant::create(compute_type, ov::Shape{}, {0.0f});
            softmax_out = std::make_shared<v1::Select>(row_valid, softmax_out, zero);
        }
    }

    ov::Output<ov::Node> qk_debug_output;
    switch (qk_matmul_output_mode) {
    case 0:
        qk_debug_output = scaled_qk->output(0);
        break;
    case 1:
        qk_debug_output = capped->output(0);
        break;
    case 2:
        qk_debug_output = masked->output(0);
        break;
    case 3:
        qk_debug_output = softmax_out->output(0);
        break;
    default:
        break;
    }

    // 6. softmax @ V
    auto output = std::make_shared<v0::MatMul>(softmax_out, V);

    return {output, qk_debug_output};
}

}  // namespace attention
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
