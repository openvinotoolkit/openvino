// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note PagedSelectiveSSM op class is under development and subject to change
///
/// \brief Operator performing paged SelectiveSSM computation for continuous batching.
///
/// Paged variant of the SelectiveSSM (Mamba2 selective state-space, arXiv:2405.21060) recurrence. Processes
/// tokens from multiple sequences packed into a single batch and manages the recurrent SSM state (a per-head
/// ``[head_dim, state_size]`` matrix) using a paged block table, enabling non-contiguous memory allocation
/// across sequences. Per token: ``dA = exp(A * dt)``, ``dtB = dt * B``, ``dBx = dtB (x) x``,
/// ``state = state * dA + dBx``, ``output = sum(state * C)``.
///
/// ``B`` and ``C`` are grouped and shared across heads: each head ``h`` reads group
/// ``g = h / heads_per_group``, where ``heads_per_group = num_heads / num_groups``.
///
/// ``recurrent_state_table`` is updated in place. For sequence ``s``, the state is cached every
/// ``cache_interval[s]`` tokens into the blocks addressed by
/// ``la_block_indices[la_block_indices_begins[s] : la_block_indices_begins[s+1]]``;
/// ``cache_interval[s] <= 0`` disables caching for that sequence. ``num_processed_tokens[s]`` gives the
/// count of previously processed tokens, used to resume from the correct cached state and block offset.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedSelectiveSSM : public ov::op::Op {
public:
    OPENVINO_OP("PagedSelectiveSSM");

    PagedSelectiveSSM() = default;
    /// \brief Constructs a PagedSelectiveSSM operation.
    ///
    /// \param A (Negative) log-decay rates per head [num_heads].
    /// \param dt Per-token, per-head time steps used for discretization [batch_size_in_tokens, num_heads].
    /// \param B Grouped input projection [batch_size_in_tokens, num_groups, state_size].
    /// \param x Input hidden states [batch_size_in_tokens, num_heads, head_dim].
    /// \param C Grouped output projection [batch_size_in_tokens, num_groups, state_size].
    /// \param recurrent_state_table Paged table of recurrent state snapshots, updated in place
    ///        [num_blocks, num_heads, head_dim, state_size]; an all-zeros tensor before any tokens are cached.
    /// \param subsequence_begins Start indices of each sequence's tokens in the flattened token batch
    ///        [batch_size_in_sequences + 1], element type i32 or i64.
    /// \param la_block_indices Physical block row indices into recurrent_state_table, concatenated across
    ///        all sequences [num_blocks], element type i32 or i64.
    /// \param la_block_indices_begins Splits la_block_indices among sequences
    ///        [batch_size_in_sequences + 1], element type i32 or i64.
    /// \param num_processed_tokens Number of tokens already processed for each sequence
    ///        [batch_size_in_sequences], element type i32 or i64.
    /// \param cache_interval Interval (in tokens) at which the recurrent state is cached for each sequence;
    ///        a value <= 0 disables caching [batch_size_in_sequences], element type i32 or i64.
    PagedSelectiveSSM(const Output<Node>& A,
                      const Output<Node>& dt,
                      const Output<Node>& B,
                      const Output<Node>& x,
                      const Output<Node>& C,
                      const Output<Node>& recurrent_state_table,
                      const Output<Node>& subsequence_begins,
                      const Output<Node>& la_block_indices,
                      const Output<Node>& la_block_indices_begins,
                      const Output<Node>& num_processed_tokens,
                      const Output<Node>& cache_interval);

    /// \brief Constructs a PagedSelectiveSSM operation from input vector.
    ///
    /// \param args Input tensor vector (11 inputs in order listed above).
    explicit PagedSelectiveSSM(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
