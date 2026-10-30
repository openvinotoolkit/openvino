// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v17 {
/// \brief SparseAttention - Scaled dot-product attention computed only over the key blocks selected for each query.
///
/// Computes grouped-query scaled dot-product attention in which every query token attends only to
/// the tokens of its selected key blocks, instead of to the whole key sequence. Key blocks are
/// groups of `key_block_size` consecutive key tokens; the ones a query attends to are listed in
/// `block_indices` and, in the causal mode, the incomplete block containing the query is added.
/// Query tokens share a selection in groups of `query_block_size` tokens. The selected blocks
/// differ per query token and are normally produced by SparseAttentionIndexer.
/// The optional `sinks` hold one learnable logit per query head.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SparseAttention : public Op {
public:
    OPENVINO_OP("SparseAttention", "opset17", Op);

    SparseAttention() = default;

    /// \brief Constructs a SparseAttention operation from an ordered list of inputs.
    ///
    /// \param inputs query, key, value, block_indices, block_count and optionally key_length and scale
    /// \param key_block_size Number of consecutive key tokens forming one key block
    /// \param query_block_size Number of consecutive query tokens sharing one selection
    /// \param causal Whether the query tokens are the last key tokens and attend to their incomplete block
    SparseAttention(const OutputVector& inputs,
                    int64_t key_block_size,
                    int64_t query_block_size = 1,
                    bool causal = true);

    /// \brief Constructs a SparseAttention operation without padding and with the default scale.
    ///
    /// \param query Attention query of shape [N, H, L, E]
    /// \param key Attention key of shape [N, Hkv, S, E]
    /// \param value Attention value of shape [N, Hkv, S, Ev]
    /// \param block_indices Selected key blocks of shape [N, Hs, Lq, KB]
    /// \param block_count Number of valid entries of every row of block_indices, shape [N, Hs, Lq]
    SparseAttention(const Output<Node>& query,
                    const Output<Node>& key,
                    const Output<Node>& value,
                    const Output<Node>& block_indices,
                    const Output<Node>& block_count,
                    int64_t key_block_size,
                    int64_t query_block_size = 1,
                    bool causal = true);

    /// \brief Constructs a SparseAttention operation for a batch with padding.
    ///
    /// \param key_length Number of valid key tokens of every sequence, shape [N] or [1]
    SparseAttention(const Output<Node>& query,
                    const Output<Node>& key,
                    const Output<Node>& value,
                    const Output<Node>& block_indices,
                    const Output<Node>& block_count,
                    const Output<Node>& key_length,
                    int64_t key_block_size,
                    int64_t query_block_size = 1,
                    bool causal = true);

    /// \brief Constructs a SparseAttention operation with an explicit attention scale.
    ///
    /// \param scale Attention scale factor used instead of the default 1 / sqrt(E)
    SparseAttention(const Output<Node>& query,
                    const Output<Node>& key,
                    const Output<Node>& value,
                    const Output<Node>& block_indices,
                    const Output<Node>& block_count,
                    const Output<Node>& key_length,
                    const Output<Node>& scale,
                    int64_t key_block_size,
                    int64_t query_block_size = 1,
                    bool causal = true);

    /// \brief Constructs a SparseAttention operation with attention sinks.
    ///
    /// \param sinks Attention sink logit of every query head, shape [H]
    SparseAttention(const Output<Node>& query,
                    const Output<Node>& key,
                    const Output<Node>& value,
                    const Output<Node>& block_indices,
                    const Output<Node>& block_count,
                    const Output<Node>& key_length,
                    const Output<Node>& scale,
                    const Output<Node>& sinks,
                    int64_t key_block_size,
                    int64_t query_block_size = 1,
                    bool causal = true);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_key_block_size() const {
        return m_key_block_size;
    }
    void set_key_block_size(int64_t key_block_size) {
        m_key_block_size = key_block_size;
    }

    int64_t get_query_block_size() const {
        return m_query_block_size;
    }
    void set_query_block_size(int64_t query_block_size) {
        m_query_block_size = query_block_size;
    }

    bool get_causal() const {
        return m_causal;
    }
    void set_causal(bool causal) {
        m_causal = causal;
    }

private:
    int64_t m_key_block_size = 0;
    int64_t m_query_block_size = 1;
    bool m_causal = true;
};
}  // namespace ov::op::v17
