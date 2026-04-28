// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note PagedCausalConv1D op class is under development and subject to change
///
/// \brief Operator performing paged causal 1D convolution for linear attention models.
///
/// Computes causal convolution over input embeddings using a block-based conv state table
/// managed by the paged execution pipeline. The conv state table is updated in-place by the kernel.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedCausalConv1D : public ov::op::Op {
public:
    OPENVINO_OP("PagedCausalConv1D");

    PagedCausalConv1D() = default;
    /// \brief Constructs a PagedCausalConv1D operation.
    ///
    /// \param input_embeds Input embeddings [batch_size_in_tokens, hidden_size].
    /// \param conv_state_table Block table containing conv_cache states [num_blocks, hidden_size, kernel_size].
    /// \param conv_weight Convolution weight [out_channels, hidden_size/group_size, conv_kernel_size].
    /// \param conv_bias Convolution bias [out_channels] or [0] (empty = no bias).
    /// \param subsequence_begins Start indices of tokens from current sequences [batch_size_in_sequences+1],
    ///        element type i32 or i64.
    /// \param la_block_indices Block index along 0-th dim in conv_state table [num_blocks],
    ///        element type i32 or i64.
    /// \param la_block_indices_begins Defines how block indices are split among sequences [batch_size_in_sequences+1],
    ///        element type i32 or i64.
    /// \param processed_tokens Number of tokens already handled per sequence [batch_size_in_sequences],
    ///        element type i32 or i64.
    /// \param cache_interval Interval between tokens to cache conv_state [batch_size_in_sequences],
    ///        element type i32 or i64.
    PagedCausalConv1D(const Output<Node>& input_embeds,
                      const Output<Node>& conv_state_table,
                      const Output<Node>& conv_weight,
                      const Output<Node>& conv_bias,
                      const Output<Node>& subsequence_begins,
                      const Output<Node>& la_block_indices,
                      const Output<Node>& la_block_indices_begins,
                      const Output<Node>& processed_tokens,
                      const Output<Node>& cache_interval);

    /// \brief Constructs a PagedCausalConv1D operation from input vector.
    ///
    /// \param args Input tensor vector (9 inputs in order listed above).
    PagedCausalConv1D(const ov::OutputVector& args);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::op::internal
