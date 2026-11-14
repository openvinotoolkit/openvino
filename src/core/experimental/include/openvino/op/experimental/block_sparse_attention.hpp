// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::experimental::v1 {

/// \note BlockSparseAttention op class is under development and subject to change
///
/// \brief Scaled dot-product attention restricted to a sparsity pattern given at block
///        granularity, as used by FlashVSR's `block_sparse_attn_func`.
///
/// The `Lq x Lk` score matrix is tiled into `block_size_q x block_size_kv` blocks; only
/// blocks marked `true` in `block_mask` are attended to (masked blocks are dropped from
/// the row-wise softmax entirely, mathematically equivalent to a token-level mask of
/// `-inf` but without materializing it).
///
/// Query/key/value use a heads-inner, token-major layout `[B, L, H, D]` (unlike the public
/// `ScaledDotProductAttention`'s `[B, H, L, D]`) to match the layout produced by the
/// windowing/mask-generation ops this op is designed to sit next to (see `DraftBlockMask`).
/// `block_mask` keeps the head dimension before the block dims
/// (`[Bm, Hm, num_q_blocks, num_kv_blocks]`) regardless of that choice.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API BlockSparseAttention : public ov::op::Op {
public:
    OPENVINO_OP("BlockSparseAttention", "experimental_opset1");

    BlockSparseAttention() = default;

    /// \brief Constructs a BlockSparseAttention op without an explicit scale input
    ///        (defaults to `1/sqrt(E)`).
    ///
    /// \param query          [B, Lq, Hq, E]
    /// \param key            [B, Lk, Hkv, E], `Hq % Hkv == 0` (GQA, like SDPA).
    /// \param value          [B, Lk, Hkv, Ev]
    /// \param block_mask     [Bm, Hm, ceil(Lq/block_size_q), ceil(Lk/block_size_kv)], boolean;
    ///                       `Bm` in {1, B}, `Hm` in {1, Hkv, Hq}.
    /// \param block_size_q   Query block size, > 0.
    /// \param block_size_kv  Key/value block size, > 0.
    /// \param causal         Token-level causal masking, ANDed with `block_mask`.
    BlockSparseAttention(const Output<Node>& query,
                         const Output<Node>& key,
                         const Output<Node>& value,
                         const Output<Node>& block_mask,
                         int64_t block_size_q,
                         int64_t block_size_kv,
                         bool causal = false);

    /// \brief Constructs a BlockSparseAttention op with an explicit scale input.
    ///
    /// \param scale  Scalar softmax scale, overrides the `1/sqrt(E)` default.
    BlockSparseAttention(const Output<Node>& query,
                         const Output<Node>& key,
                         const Output<Node>& value,
                         const Output<Node>& block_mask,
                         const Output<Node>& scale,
                         int64_t block_size_q,
                         int64_t block_size_kv,
                         bool causal = false);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool can_constant_fold(const OutputVector&) const override {
        return false;
    }

    bool get_causal() const {
        return m_causal;
    }
    int64_t get_block_size_q() const {
        return m_block_size_q;
    }
    int64_t get_block_size_kv() const {
        return m_block_size_kv;
    }

private:
    bool m_causal = false;
    int64_t m_block_size_q = 0;
    int64_t m_block_size_kv = 0;
};

}  // namespace ov::op::experimental::v1

namespace ov::op::experimental {
using v1::BlockSparseAttention;
}  // namespace ov::op::experimental
