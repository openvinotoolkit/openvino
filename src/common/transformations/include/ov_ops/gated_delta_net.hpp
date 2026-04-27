// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace op {
namespace internal {

/// \brief GatedDeltaNet - linear recurrent sequence processing operation.
///
/// \note Implements the recurrence from arXiv:2412.06464. Processes a sequence using
///       the delta rule to update a hidden state matrix, controlled by a per-token
///       forget gate and a per-token write gate (beta). Queries are scaled internally
///       by 1 / sqrt(key_head_dim).
///
/// Inputs (6):
///   query           [batch, seq_len, num_heads, key_head_dim]
///   key             [batch, seq_len, num_heads, key_head_dim]
///   value           [batch, seq_len, num_heads, value_head_dim]
///   recurrent_state [batch, num_heads, key_head_dim, value_head_dim]
///   gate            [batch, seq_len, num_heads]  (log-space forget gate)
///   beta            [batch, seq_len, num_heads]  (write gate)
///
/// Outputs (2):
///   output_attn             [batch, seq_len, num_heads, value_head_dim]
///   output_recurrent_state  [batch, num_heads, key_head_dim, value_head_dim]
class TRANSFORMATIONS_API GatedDeltaNet : public ov::op::Op {
public:
    OPENVINO_OP("GatedDeltaNet", "ie_internal_opset");

    GatedDeltaNet() = default;

    /// \brief Constructs a GatedDeltaNet operation.
    ///
    /// \param query           4D tensor [batch, seq_len, num_heads, key_head_dim]
    /// \param key             4D tensor [batch, seq_len, num_heads, key_head_dim]
    /// \param value           4D tensor [batch, seq_len, num_heads, value_head_dim]
    /// \param recurrent_state 4D tensor [batch, num_heads, key_head_dim, value_head_dim]
    /// \param gate            3D tensor [batch, seq_len, num_heads] (log-space)
    /// \param beta            3D tensor [batch, seq_len, num_heads]
    GatedDeltaNet(const Output<Node>& query,
                  const Output<Node>& key,
                  const Output<Node>& value,
                  const Output<Node>& recurrent_state,
                  const Output<Node>& gate,
                  const Output<Node>& beta);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
};

}  // namespace internal
}  // namespace op
}  // namespace ov
