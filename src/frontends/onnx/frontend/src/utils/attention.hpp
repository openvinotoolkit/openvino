// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

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

/// \brief Convert boolean mask to float additive mask: true -> 0.0, false -> mask_filter_value.
///
/// \param mask              Boolean tensor where true means "attend" and false means "mask out".
/// \param type              Target floating-point element type for the output mask.
/// \param mask_filter_value Value assigned to masked (false) positions (default: -10000.0).
/// \return A Select node producing an additive float mask of the same shape as the input.
ov::Output<ov::Node> convert_boolean_mask(const ov::Output<ov::Node>& mask,
                                          const ov::element::Type& type,
                                          float mask_filter_value = -10000.0f);

/// \brief Build additive causal mask of shape (seq_q, seq_kv): 0 for allowed, -10000 for masked.
///
/// \param Q          Query tensor of shape (B, heads, seq_q, head_size).
/// \param K          Key tensor of shape (B, heads, seq_kv, head_size).
/// \param use_offset When true, adjusts row indices for KV cache offset so that query position i
///                   attends to key positions j where j <= i + (seq_kv - seq_q). When false, builds
///                   a simple lower-triangular mask (np.tril with k=0).
/// \return A float additive mask broadcastable over the attention scores.
ov::Output<ov::Node> build_causal_mask(const ov::Output<ov::Node>& Q, const ov::Output<ov::Node>& K, bool use_offset);

/// \brief Build SDPA-based attention (primary fast path).
///
/// \param Q          Query tensor of shape (B, heads, seq_q, head_size).
/// \param K          Key tensor of shape (B, heads, seq_kv, head_size).
/// \param V          Value tensor of shape (B, heads, seq_kv, head_size).
/// \param has_mask   Whether an attention mask is provided.
/// \param attn_mask  Additive attention mask (used only when has_mask is true).
/// \param scale_attr Explicit scale factor; when 0.0 the SDPA op uses the default 1/sqrt(head_size).
/// \param is_causal  When true, the SDPA op applies internal causal masking.
/// \return The output of ScaledDotProductAttention.
ov::Output<ov::Node> build_sdpa(const ov::Output<ov::Node>& Q,
                                const ov::Output<ov::Node>& K,
                                const ov::Output<ov::Node>& V,
                                bool has_mask,
                                const ov::Output<ov::Node>& attn_mask,
                                float scale_attr,
                                bool is_causal);

/// \brief Build manual attention decomposition (for softcap or qk_matmul_output).
///
/// \param Q                     Query tensor of shape (B, heads, seq_q, head_size).
/// \param K                     Key tensor of shape (B, heads, seq_kv, head_size).
/// \param V                     Value tensor of shape (B, heads, seq_kv, head_size).
/// \param has_mask              Whether an attention mask is provided.
/// \param attn_mask             Additive attention mask (used only when has_mask is true).
/// \param scale_attr            Explicit scale factor; when 0.0, uses default 1/sqrt(head_size).
/// \param softcap               Soft-capping value; when > 0, applies softcap * tanh(scores / softcap).
/// \param is_causal             When true, adds an additive causal mask to attention scores.
/// \param qk_matmul_output_mode Selects at which stage the intermediate QK tensor is captured:
///                              0 = after scale, 1 = after mask, 2 = after softcap, 3 = after softmax.
/// \param needs_qk_output       When true, the intermediate QK tensor is returned as the second output.
/// \return A two-element OutputVector: {attention_output, qk_intermediate_or_empty}.
ov::OutputVector build_manual_attention(const ov::Output<ov::Node>& Q,
                                        const ov::Output<ov::Node>& K,
                                        const ov::Output<ov::Node>& V,
                                        bool has_mask,
                                        const ov::Output<ov::Node>& attn_mask,
                                        float scale_attr,
                                        float softcap,
                                        bool is_causal,
                                        int64_t qk_matmul_output_mode,
                                        bool needs_qk_output);

}  // namespace attention
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
