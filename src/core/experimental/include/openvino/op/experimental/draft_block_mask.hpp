// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::experimental::v1 {

/// \note DraftBlockMask op class is under development and subject to change
///
/// \brief Builds a block-level top-k sparse-attention mask from mean-pooled query/key
///        blocks plus a precomputed static local-window bias, as used by FlashVSR's
///        draft attention path.
///
/// Reference semantics (per attention head h, per query block row):
///   q_h = mean(q_windows, axis=1) reshaped/split into heads         -> (H, Nq, head_dim)
///   k_h = mean(k_windows, axis=1) reshaped/split into heads         -> (H, Nk, head_dim)
///   scores = matmul(q_h, k_h^T) / sqrt(head_dim) + local_mask_bias  -> (H, Nq, Nk)
///   attn   = softmax(scores, axis=-1)
///   mask   = attn > kth_largest(attn per (head, temporal-block) row, topk)
///   out    = tile(mask, batch_size)                                 -> (B, H, Nq, Nk)
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API DraftBlockMask : public ov::op::Op {
public:
    OPENVINO_OP("DraftBlockMask", "experimental_opset1");

    DraftBlockMask() = default;

    /// \brief Constructs a DraftBlockMask operation.
    ///
    /// \param q_windows   Windowed query blocks, shape [Nq, block_s, D].
    /// \param k_windows   Windowed key blocks, shape [Nk, block_s, D].
    /// \param local_mask  Precomputed static sliding-window bias mask, boolean, shape [HW, HW]
    ///                    (always square; HW is local_mask's own shape, not derived from Nq/Nk).
    ///                    Constrains Nq == seqlen * HW exactly. Nk need only be a multiple of HW --
    ///                    the K-side temporal group count (Nk / HW) is independent of seqlen (e.g. a
    ///                    streaming decoder's accumulated key/value cache growing past the current
    ///                    query window). Not recomputed by this op.
    /// \param num_heads   Number of attention heads; must evenly divide D.
    /// \param seqlen      Number of temporal blocks in query windows (Nq = seqlen * HW).
    /// \param topk        Number of key blocks kept per (head, temporal-block) group of HW query
    ///                    rows, not an independent budget per row.
    /// \param batch_size  Output batch dimension. `q_windows`/`k_windows` carry no batch axis, so
    ///                    there is structurally only one mask to compute; this only controls how many
    ///                    times that single mask is tiled into the output (see `out = tile(...)`
    ///                    above), not an independent per-batch computation.
    DraftBlockMask(const Output<Node>& q_windows,
                   const Output<Node>& k_windows,
                   const Output<Node>& local_mask,
                   int64_t num_heads,
                   int64_t seqlen,
                   int64_t topk,
                   int64_t batch_size = 1);

    /// \brief Constructs a DraftBlockMask operation with runtime `seqlen`/`topk` inputs, for use in a
    ///        graph where those values are only known at inference time (one compiled graph serving
    ///        multiple chunk shapes/resolutions) -- see the attribute-based constructor above for the
    ///        fixed-shape case. `q_windows`/`k_windows`/`local_mask` may all be dynamically shaped.
    ///
    /// \param seqlen  Scalar (rank 0 or 1) i32/i64 -- number of temporal blocks in query windows.
    /// \param topk    Scalar (rank 0 or 1) i32/i64 -- number of key blocks kept per (head,
    ///                temporal-block) group of HW query rows.
    DraftBlockMask(const Output<Node>& q_windows,
                   const Output<Node>& k_windows,
                   const Output<Node>& local_mask,
                   const Output<Node>& seqlen,
                   const Output<Node>& topk,
                   int64_t num_heads,
                   int64_t batch_size = 1);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    bool can_constant_fold(const OutputVector&) const override {
        return false;
    }

    int64_t get_num_heads() const {
        return m_num_heads;
    }
    /// \note Returns the `-1` sentinel when constructed via the runtime-input (5-input) constructor.
    int64_t get_seqlen() const {
        return m_seqlen;
    }
    /// \note Returns the `-1` sentinel when constructed via the runtime-input (5-input) constructor.
    int64_t get_topk() const {
        return m_topk;
    }
    int64_t get_batch_size() const {
        return m_batch_size;
    }

private:
    int64_t m_num_heads = 0;
    int64_t m_seqlen = 0;
    int64_t m_topk = 0;
    int64_t m_batch_size = 1;
};

}  // namespace ov::op::experimental::v1

namespace ov::op::experimental {
using v1::DraftBlockMask;
}  // namespace ov::op::experimental
