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
/// Implements the time-sequential single-step recurrence used by Mamba2 mixers in hybrid
/// Mamba2 models such as NemotronH (see arXiv:2405.21060). The discretized inputs `dA`,
/// `dBx` and `C` are precomputed and vectorized over the sequence outside of this op; the
/// operation only performs the state recurrence and the per-token readout:
///     state_t = state_{t-1} * dA_t + dBx_t
///     y_t     = reduce_sum(state_t * C_t, axis=state_size)
/// \ingroup ov_ops_cpp_api
class OPENVINO_API Mamba2 : public ov::op::Op {
public:
    OPENVINO_OP("Mamba2");

    Mamba2() = default;
    /// \brief Constructs a Mamba2 operation.
    ///
    /// \param dA Discretized state transition tensor of shape [batch, num_heads, seq_len, 1, 1].
    /// \param dBx Discretized input contribution (B * x) of shape
    ///        [batch, num_heads, seq_len, head_dim, state_size].
    /// \param C Per-token output projection of shape [batch, num_heads, seq_len, state_size].
    /// \param recurrent_state Initial SSM hidden state of shape
    ///        [batch, num_heads, head_dim, state_size].
    Mamba2(const Output<Node>& dA, const Output<Node>& dBx, const Output<Node>& C, const Output<Node>& recurrent_state);

    /// \brief Constructs a Mamba2 operation from an input vector.
    ///
    /// \param args Input tensor vector in order: dA, dBx, C, recurrent_state.
    explicit Mamba2(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
