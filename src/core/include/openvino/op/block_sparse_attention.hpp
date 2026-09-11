// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v17 {
/// \brief Block-sparse scaled dot product attention.
///
/// Computes exact scaled dot-product attention restricted to a caller-supplied,
/// per-query-block selection of key/value blocks:
///
///     output = softmax(scale * query @ gather(key, block_indices)^T + causal_bias)
///                 @ gather(value, block_indices)
///
/// The gather of key/value blocks happens *inside* the kernel: the operation reads
/// directly from the original (non-gathered, non-transposed) key/value tensors using
/// `block_indices`, so no auxiliary Gather/Transpose/Reshape subgraph is needed around
/// it. This is the key difference versus decomposing the same computation with existing
/// ops (Gather + Reshape + ScaledDotProductAttention): that decomposition requires
/// materializing a gathered-and-transposed copy of key/value before the attention
/// matmuls, whose cost scales with `batch * heads * num_query_blocks * gathered_length`
/// and can dominate at small sequence lengths. BlockSparseAttention avoids that
/// materialization entirely, so its cost scales with the actual amount of selected
/// (sparse) work regardless of sequence length.
///
/// Block selection itself (pooling / coarse scoring / top-k) is expected to be built
/// from existing ops (ReduceMean, MatMul, TopK, ...) upstream in the graph and passed in
/// as `block_indices`; this operation only fuses "gather selected blocks" with "attend".
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BlockSparseAttention : public Op {
public:
    OPENVINO_OP("BlockSparseAttention", "opset17", ov::op::Op);

    BlockSparseAttention() = default;

    /// \brief Constructs a BlockSparseAttention operation from a flat input list.
    ///
    /// \param inputs     3 to 6 inputs: query, key, value, block_indices,
    ///                   [block_indices_mask], [scale]. See the named-argument
    ///                   constructors below for the meaning of each input.
    /// \param block_size Number of contiguous key/value tokens per block.
    /// \param causal     Apply an additional token-level causal mask.
    BlockSparseAttention(const OutputVector& inputs, int64_t block_size, bool causal = false);

    /// \brief Constructs a BlockSparseAttention operation with an explicit scale.
    ///
    /// \param query               Query tensor, shape `[B, H, L, E]`.
    /// \param key                 Key tensor, shape `[B, Hk, S, E]`. `Hk` must be either `H`
    ///                            (one key head per query head) or `1` (a single key head
    ///                            broadcast to every query head), matching the level of head
    ///                            broadcasting ScaledDotProductAttention itself supports.
    /// \param value               Value tensor, shape `[B, Hk, S, Ev]` (same `Hk` as `key`).
    /// \param block_indices       Indices, into the `S / block_size` key/value blocks,
    ///                            of the blocks selected for each query block. Shape
    ///                            `[B, Hb, L / block_size, k_blocks]`, integer type, `Hb`
    ///                            either `H` or `1` (broadcast), same rule as `Hk` above.
    /// \param block_indices_mask  Boolean tensor with the same shape as `block_indices`;
    ///                            `false` marks a padding entry that must not contribute
    ///                            to the output (used when the number of true candidate
    ///                            blocks is ragged across query blocks).
    /// \param scale               Custom softmax scale (scalar, or single-element
    ///                            tensor). Defaults to `1 / sqrt(E)` when omitted.
    /// \param block_size          Number of contiguous key/value tokens per block.
    /// \param causal              Apply an additional token-level causal mask.
    BlockSparseAttention(const Output<Node>& query,
                         const Output<Node>& key,
                         const Output<Node>& value,
                         const Output<Node>& block_indices,
                         const Output<Node>& block_indices_mask,
                         const Output<Node>& scale,
                         int64_t block_size,
                         bool causal = false);

    /// \brief Constructs a BlockSparseAttention operation without an explicit scale.
    BlockSparseAttention(const Output<Node>& query,
                         const Output<Node>& key,
                         const Output<Node>& value,
                         const Output<Node>& block_indices,
                         const Output<Node>& block_indices_mask,
                         int64_t block_size,
                         bool causal = false);

    /// \brief Constructs a BlockSparseAttention operation without a mask or scale.
    BlockSparseAttention(const Output<Node>& query,
                         const Output<Node>& key,
                         const Output<Node>& value,
                         const Output<Node>& block_indices,
                         int64_t block_size,
                         bool causal = false);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    int64_t get_block_size() const {
        return m_block_size;
    }

    void set_block_size(int64_t block_size) {
        m_block_size = block_size;
    }

    bool get_causal() const {
        return m_causal;
    }

    void set_causal(bool causal) {
        m_causal = causal;
    }

private:
    int64_t m_block_size = 0;
    bool m_causal = false;
};

}  // namespace ov::op::v17
