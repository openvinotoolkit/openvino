// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note SelectiveSSM op class is under development and subject to change
///
/// \brief Operator performing the Mamba2 selective state-space recurrence (arXiv:2405.21060).
///
/// Discretizes ``A`` (log-decay rates) and ``B`` (input projection) with ``dt`` (time steps) ahead of the
/// recurrence: ``dA = exp(A * dt)``, ``dtB = dt * B``. Then, for each token ``t``:
/// ``dBx_t = x_t (x) dtB_t``, ``state_t = state_{t-1} * dA_t + dBx_t``, ``y_t = sum(state_t * C_t)``.
///
/// ``B`` and ``C`` are grouped and shared across heads: each head ``h`` reads group
/// ``g = h / heads_per_group``, where ``heads_per_group = num_heads / num_groups``.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SelectiveSSM : public ov::op::Op {
public:
    OPENVINO_OP("SelectiveSSM");

    SelectiveSSM() = default;
    /// \brief Constructs a SelectiveSSM operation.
    ///
    /// \param A (Negative) log-decay rates per head [num_heads].
    /// \param dt Per-token, per-head time steps used for discretization [batch_size, seq_len, num_heads].
    /// \param B Grouped input projection [batch_size, seq_len, num_groups, state_size].
    /// \param x Input hidden states [batch_size, seq_len, num_heads, head_dim].
    /// \param C Grouped output projection [batch_size, seq_len, num_groups, state_size].
    /// \param recurrent_state Initial SSM hidden state [batch_size, num_heads, head_dim, state_size];
    ///        an all-zeros tensor for a fresh sequence.
    SelectiveSSM(const Output<Node>& A,
                 const Output<Node>& dt,
                 const Output<Node>& B,
                 const Output<Node>& x,
                 const Output<Node>& C,
                 const Output<Node>& recurrent_state);

    /// \brief Constructs a SelectiveSSM operation from input vector.
    ///
    /// \param args Input tensor vector in order: A, dt, B, x, C, recurrent_state.
    explicit SelectiveSSM(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
