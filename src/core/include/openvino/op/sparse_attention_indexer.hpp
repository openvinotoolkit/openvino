// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v17 {
/// \brief Sparse attention indexer - picks the key blocks that SparseAttention attends to.
///
/// For every query token the operation scores the complete key blocks visible to it with the
/// multi-head, ReLU-rectified dot product of the index query heads and the block index key, and
/// returns the indices of the `token_budget / key_block_size` best-scoring blocks together with
/// their number. These indices are the sparsity pattern consumed by SparseAttention, recomputed
/// per query token instead of being fixed.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SparseAttentionIndexer : public Op {
public:
    OPENVINO_OP("SparseAttentionIndexer", "opset17", Op);

    SparseAttentionIndexer() = default;

    /// \brief Constructs a SparseAttentionIndexer operation from an ordered list of inputs.
    ///
    /// \param inputs index_query, block_index_keys, key_length and optionally index_weights
    /// \param key_block_size Number of consecutive key tokens forming one key block
    /// \param token_budget Maximum number of tokens selected from complete blocks for one query
    /// \param num_selection_heads Number of selections per query token
    SparseAttentionIndexer(const OutputVector& inputs,
                           int64_t key_block_size,
                           int64_t token_budget,
                           int64_t num_selection_heads = 1);

    /// \brief Constructs a SparseAttentionIndexer operation with unit index weights.
    ///
    /// \param index_query Index query heads of shape [N, Hi, L, Di]
    /// \param block_index_keys One index key per complete key block, shape [N, B, Di]
    /// \param key_length Number of valid key tokens of every sequence, shape [N] or [1]
    SparseAttentionIndexer(const Output<Node>& index_query,
                           const Output<Node>& block_index_keys,
                           const Output<Node>& key_length,
                           int64_t key_block_size,
                           int64_t token_budget,
                           int64_t num_selection_heads = 1);

    /// \brief Constructs a SparseAttentionIndexer operation with per-head index weights.
    ///
    /// \param index_weights Weight of every index query head for every query token, shape [N, L, Hi]
    SparseAttentionIndexer(const Output<Node>& index_query,
                           const Output<Node>& block_index_keys,
                           const Output<Node>& key_length,
                           const Output<Node>& index_weights,
                           int64_t key_block_size,
                           int64_t token_budget,
                           int64_t num_selection_heads = 1);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_key_block_size() const {
        return m_key_block_size;
    }
    void set_key_block_size(int64_t key_block_size) {
        m_key_block_size = key_block_size;
    }

    int64_t get_token_budget() const {
        return m_token_budget;
    }
    void set_token_budget(int64_t token_budget) {
        m_token_budget = token_budget;
    }

    int64_t get_num_selection_heads() const {
        return m_num_selection_heads;
    }
    void set_num_selection_heads(int64_t num_selection_heads) {
        m_num_selection_heads = num_selection_heads;
    }

private:
    int64_t m_key_block_size = 0;
    int64_t m_token_budget = 0;
    int64_t m_num_selection_heads = 1;
};
}  // namespace ov::op::v17
