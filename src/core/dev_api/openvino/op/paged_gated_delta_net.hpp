// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::internal {
/// \note PagedGatedDeltaNet op class is under development and subject to change
///
/// \brief Operator performing paged Gated Delta Net computation for linear attention models.
///
/// Computes gated delta net attention over input tokens using a block-based recurrent state table
/// managed by the paged execution pipeline. The recurrent state table is updated in-place by the kernel.
///
/// This operation uses grouped-query linear attention. The number of groups is
/// ``num_groups = v_num_heads / num_heads``. Each query and key head is shared by
/// ``num_groups`` consecutive value heads, with the mapping ``h_q = h_v / num_groups``.
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedGatedDeltaNet : public ov::op::Op {
public:
    OPENVINO_OP("PagedGatedDeltaNet");

    PagedGatedDeltaNet() = default;
    /// \brief Constructs a PagedGatedDeltaNet operation.
    ///
    /// \param query Query tensor [batch_size_in_tokens, num_heads, key_head_dim].
    ///        num_heads is the number of query/key heads; v_num_heads must be divisible by num_heads.
    /// \param key Key tensor [batch_size_in_tokens, num_heads, key_head_dim].
    /// \param value Value tensor [batch_size_in_tokens, v_num_heads, value_head_dim].
    ///        v_num_heads >= num_heads; the ratio v_num_heads/num_heads gives the number of GQA groups.
    /// \param recurrent_state_table Block table containing recurrent states
    ///        [num_blocks, v_num_heads, value_head_dim, key_head_dim].
    /// \param gate Gate tensor [batch_size_in_tokens, v_num_heads].
    /// \param beta Beta tensor [batch_size_in_tokens, v_num_heads].
    /// \param subsequence_begins Start indices of tokens from current sequences [batch_size_in_sequences+1],
    ///        element type i32 or i64.
    /// \param la_block_indices Block index along 0-th dim in recurrent_state table [num_blocks],
    ///        element type i32 or i64.
    /// \param la_block_indices_begins Defines how block indices are split among sequences [batch_size_in_sequences+1],
    ///        element type i32 or i64.
    /// \param processed_tokens Number of tokens already handled per sequence [batch_size_in_sequences],
    ///        element type i32 or i64.
    /// \param cache_interval Interval between tokens to cache state [batch_size_in_sequences],
    ///        element type i32 or i64.
    /// \param use_qk_l2norm Enables q/k L2-normalization inside this op.
    /// \param q_l2_norm_eps Positive floating-point epsilon used for query L2-normalization.
    /// \param k_l2_norm_eps Positive floating-point epsilon used for key L2-normalization.
    PagedGatedDeltaNet(const Output<Node>& query,
                       const Output<Node>& key,
                       const Output<Node>& value,
                       const Output<Node>& recurrent_state_table,
                       const Output<Node>& gate,
                       const Output<Node>& beta,
                       const Output<Node>& subsequence_begins,
                       const Output<Node>& la_block_indices,
                       const Output<Node>& la_block_indices_begins,
                       const Output<Node>& processed_tokens,
                       const Output<Node>& cache_interval,
                       bool use_qk_l2norm = false,
                       float q_l2_norm_eps = 1e-6F,
                       float k_l2_norm_eps = 1e-6F);

    /// \brief Constructs a PagedGatedDeltaNet operation from input vector.
    ///
    /// \param args Input tensor vector (11 inputs in order listed above).
    /// \param use_qk_l2norm Enables q/k L2-normalization inside this op.
    /// \param q_l2_norm_eps Positive floating-point epsilon used for query L2-normalization.
    /// \param k_l2_norm_eps Positive floating-point epsilon used for key L2-normalization.
    PagedGatedDeltaNet(const ov::OutputVector& args,
                       bool use_qk_l2norm = false,
                       float q_l2_norm_eps = 1e-6F,
                       float k_l2_norm_eps = 1e-6F);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool get_use_qk_l2norm() const {
        return m_use_qk_l2norm;
    }
    float get_q_l2_norm_eps() const {
        return m_q_l2_norm_eps;
    }
    float get_k_l2_norm_eps() const {
        return m_k_l2_norm_eps;
    }

private:
    bool m_use_qk_l2norm = false;
    float m_q_l2_norm_eps = 1e-6F;
    float m_k_l2_norm_eps = 1e-6F;
};

}  // namespace ov::op::internal
