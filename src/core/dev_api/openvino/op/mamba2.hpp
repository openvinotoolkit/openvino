// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note Mamba2 op class is under development and subject to change
///
/// \brief Operator performing the Mamba2 selective state-space model (SSM) recurrence.
///
/// Implements the time-sequential recurrence used by Mamba2 mixers in hybrid Mamba2 models
/// such as NemotronH (see arXiv:2405.21060). The operation consumes the raw, time-major
/// projections and performs the discretization (exp, outer product), the state scan and the
/// per-token readout internally (H = num_heads, G = num_groups, P = head_dim, N = state_size):
///     dA_t    = exp(A * dt_t)
///     dBx_t   = (dt_t * B_t) outer x_t
///     state_t = state_{t-1} * dA_t + dBx_t
///     y_t     = reduce_sum(state_t * C_t, axis=state_size)
/// The `B` and `C` matrices are provided per group and broadcast to the `H` heads inside the op.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Mamba2 : public ov::op::Op {
public:
    OPENVINO_OP("Mamba2");

    Mamba2() = default;
    /// \brief Constructs a Mamba2 operation.
    ///
    /// \param A Per-head negative log-decay rates of shape [num_heads].
    /// \param dt Per-token time steps of shape [batch, seq_len, num_heads].
    /// \param B Per-group input matrix of shape [batch, seq_len, num_groups, state_size].
    /// \param x Input hidden states of shape [batch, seq_len, num_heads, head_dim].
    /// \param C Per-group output matrix of shape [batch, seq_len, num_groups, state_size].
    /// \param recurrent_state Initial SSM hidden state of shape
    ///        [batch, num_heads, head_dim, state_size].
    Mamba2(const Output<Node>& A,
           const Output<Node>& dt,
           const Output<Node>& B,
           const Output<Node>& x,
           const Output<Node>& C,
           const Output<Node>& recurrent_state);

    /// \brief Constructs a Mamba2 operation from an input vector.
    ///
    /// \param args Input tensor vector in order: A, dt, B, x, C, recurrent_state.
    explicit Mamba2(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
