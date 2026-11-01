// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/experimental/draft_block_mask.hpp"
#include "intel_gpu/primitives/primitive.hpp"

namespace cldnn {

using DraftBlockMask = ov::op::experimental::DraftBlockMask;

/// @brief draft_block_mask primitive
/// @details Builds FlashVSR's block-sparse-attention mask: mean-pools windowed Q/K blocks, scores
/// them per head, adds a static local-window bias, softmaxes, then keeps the global top-k blocks
/// per (head, temporal-group). See ov::op::experimental::DraftBlockMask for the full contract.
struct draft_block_mask : public primitive_base<draft_block_mask> {
    enum InputIdx : size_t { Q_WINDOWS = 0, K_WINDOWS = 1, LOCAL_MASK = 2, SEQLEN = 3, TOPK = 4, COUNT = 5 };

    CLDNN_DECLARE_PRIMITIVE(draft_block_mask)

    draft_block_mask() : primitive_base("", {}) {}

    /// @brief Constructs draft_block_mask primitive / layer.
    ///
    /// @param id         An identifier of new primitive.
    /// @param inputs     A list of input primitive ids: [q_windows, k_windows, local_mask].
    /// @param num_heads  Number of attention heads.
    /// @param seqlen     Number of temporal blocks folded into Nq/Nk (Nq = seqlen * HW).
    /// @param topk       Number of key blocks kept per (head, temporal-group) row.
    /// @param batch_size Output batch dimension.
    draft_block_mask(const primitive_id& id, const std::vector<input_info>& inputs, int64_t num_heads, int64_t seqlen, int64_t topk, int64_t batch_size = 1)
        : primitive_base(id, inputs),
          num_heads(num_heads),
          seqlen(seqlen),
          topk(topk),
          batch_size(batch_size) {
        OPENVINO_ASSERT(inputs.size() == 3, "[GPU] Unexpected input number for draft_block_mask primitive: ", inputs.size());
    }

    /// @brief Constructs draft_block_mask primitive / layer with runtime seqlen/topk inputs (see
    /// ov::op::experimental::DraftBlockMask's 5-input constructor).
    ///
    /// @param id         An identifier of new primitive.
    /// @param inputs     A list of input primitive ids: [q_windows, k_windows, local_mask, seqlen, topk].
    /// @param num_heads  Number of attention heads.
    /// @param batch_size Output batch dimension.
    draft_block_mask(const primitive_id& id, const std::vector<input_info>& inputs, int64_t num_heads, int64_t batch_size = 1)
        : primitive_base(id, inputs),
          num_heads(num_heads),
          seqlen(-1),
          topk(-1),
          batch_size(batch_size) {
        OPENVINO_ASSERT(inputs.size() == 5, "[GPU] Unexpected input number for draft_block_mask primitive: ", inputs.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_heads);
        seed = hash_combine(seed, seqlen);
        seed = hash_combine(seed, topk);
        seed = hash_combine(seed, batch_size);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const draft_block_mask>(rhs);
        return num_heads == rhs_casted.num_heads && seqlen == rhs_casted.seqlen && topk == rhs_casted.topk && batch_size == rhs_casted.batch_size;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<draft_block_mask>::save(ob);
        ob << num_heads;
        ob << seqlen;
        ob << topk;
        ob << batch_size;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<draft_block_mask>::load(ib);
        ib >> num_heads;
        ib >> seqlen;
        ib >> topk;
        ib >> batch_size;
    }

    int64_t num_heads = 0;
    int64_t seqlen = 0;
    int64_t topk = 0;
    int64_t batch_size = 1;
};

// Backward-compatible alias for the (now nested) input-index enum, so existing call sites that
// spell `DraftBlockMaskInputIdx::Q_WINDOWS` keep working without qualification.
using DraftBlockMaskInputIdx = draft_block_mask::InputIdx;

}  // namespace cldnn
