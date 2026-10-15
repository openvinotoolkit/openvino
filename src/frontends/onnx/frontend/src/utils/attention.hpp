// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <limits>
#include <memory>
#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace attention {

/// \brief Extracts specific dimensions from a ShapeOf node using Gather.
///
/// \param shape  A ShapeOf node whose output contains the tensor's shape.
/// \param dims   Indices of the dimensions to extract.
/// \return A Gather node producing the selected dimension values.
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf>& shape,
                                         const std::vector<int>& dims);

/// \brief Convenience overload: computes ShapeOf for the given node, then extracts dimensions.
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node>& node, const std::vector<int>& dims);

/// \brief Reshape a 3D tensor (batch, seq, num_heads * head_size) to 4D (batch, num_heads, seq, head_size).
///
/// \param input     3D tensor of shape (batch, seq, num_heads * head_size).
/// \param num_heads Number of attention heads.
/// \return A 4D tensor of shape (batch, num_heads, seq, head_size).
ov::Output<ov::Node> reshape_3d_to_4d(const ov::Output<ov::Node>& input, int64_t num_heads);

/// \brief Reshape a 4D tensor (batch, num_heads, seq, head_size) back to 3D (batch, seq, num_heads * head_size).
///
/// \param output 4D tensor of shape (batch, num_heads, seq, head_size).
/// \return A 3D tensor of shape (batch, seq, num_heads * head_size).
ov::Output<ov::Node> reshape_4d_to_3d(const ov::Output<ov::Node>& output);

/// \brief Convert boolean mask to float additive mask: true -> 0.0, false -> mask_filter_value.
///
/// \param mask              Boolean tensor where true means "attend" and false means "mask out".
/// \param type              Target floating-point element type for the output mask.
/// \param mask_filter_value Value assigned to masked (false) positions (default: -inf, required for
///                          the fully-masked-row guard in build_manual_attention; pass a finite value
///                          such as ORT's -10000 only where that guard must stay inert).
/// \return A Select node producing an additive float mask of the same shape as the input.
ov::Output<ov::Node> convert_boolean_mask(const ov::Output<ov::Node>& mask,
                                          const ov::element::Type& type,
                                          float mask_filter_value = -std::numeric_limits<float>::infinity());

/// \brief Alignment mode for the causal mask offset.
enum class CausalKind {
    NONE,    // offset 0: query i attends key j iff j <= i
    PAST,    // offset = seq_kv - seq_q (internal KV cache, bottom-right alignment)
    NONPAD,  // per-batch offset = nonpad[b] - seq_q (external KV cache)
};

/// \brief Build an additive causal mask with opset-24 bottom-right alignment.
///
/// For CausalKind::NONPAD the offset varies per batch and the result has shape (B, 1, seq_q, seq_kv);
/// otherwise the result is (seq_q, seq_kv) and broadcasts over batch and heads.
///
/// \param Q                 Query tensor of shape (B, heads, seq_q, head_size).
/// \param K                 Key tensor of shape (B, heads, seq_kv, head_size).
/// \param kind              Offset alignment mode (NONE / PAST / NONPAD).
/// \param mask_filter_value Value assigned to masked positions (default: -inf).
/// \param nonpad            Per-batch non-pad KV length (only used for CausalKind::NONPAD).
/// \return A float additive mask broadcastable over the attention scores.
ov::Output<ov::Node> build_causal_mask(const ov::Output<ov::Node>& Q,
                                       const ov::Output<ov::Node>& K,
                                       CausalKind kind = CausalKind::NONE,
                                       float mask_filter_value = -std::numeric_limits<float>::infinity(),
                                       const ov::Output<ov::Node>& nonpad = {});

/// \brief Build manual attention decomposition following ONNX opset-24 semantics.
///
/// The mask passed in is already merged (attention + causal + padding) using -inf for disallowed
/// positions. Order of operations: scale -> softcap -> mask -> softmax.
///
/// \param Q                     Query tensor of shape (B, heads, seq_q, head_size).
/// \param K                     Key tensor of shape (B, heads, seq_kv, head_size).
/// \param V                     Value tensor of shape (B, heads, seq_kv, head_size).
/// \param attn_mask             Additive attention mask (used only when has_mask is true).
/// \param scale_attr            Explicit scale factor; when 0.0, uses default 1/sqrt(head_size).
/// \param softcap               Soft-capping value; when > 0, applies softcap * tanh(scores / softcap).
/// \param qk_matmul_output_mode Selects at which stage the intermediate QK tensor is captured:
///                              0 = after scale, 1 = after softcap, 2 = after mask, 3 = after softmax.
/// \param include_safe_softmax  When true (ONNX Attention opset-23/-24), applies the fully-masked-row
///                              guard so a row whose keys are all masked (-inf) yields a zero output row
///                              instead of NaN. Set to false for ONNX Runtime MultiHeadAttention
///                              semantics (finite mask_filter_value, no zero-row guard).
/// \return A two-element OutputVector: {attention_output, qk_intermediate_or_empty}.
ov::OutputVector build_manual_attention(const ov::Output<ov::Node>& Q,
                                        const ov::Output<ov::Node>& K,
                                        const ov::Output<ov::Node>& V,
                                        const ov::Output<ov::Node>& attn_mask,
                                        float scale_attr,
                                        float softcap,
                                        int64_t qk_matmul_output_mode,
                                        bool include_safe_softmax = true);

}  // namespace attention
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
