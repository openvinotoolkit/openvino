// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "intel_gpu/graph/program.hpp"
#include "intel_gpu/graph/topology.hpp"
#include "openvino/op/experimental/block_sparse_attention.hpp"
#include "intel_gpu/primitives/primitive.hpp"

namespace cldnn {

using BlockSparseAttention = ov::op::experimental::BlockSparseAttention;

/// @brief block_sparse_attention primitive
/// @details Scaled dot-product attention restricted to a sparsity pattern given at block
/// granularity: the Lq x Lk score matrix is tiled into block_size_q x block_size_kv blocks, and
/// only blocks marked true in block_mask are attended to. Query/key/value use a heads-inner,
/// token-major layout [B, L, H, D]. See ov::op::experimental::BlockSparseAttention for the full
/// contract.
struct block_sparse_attention : public primitive_base<block_sparse_attention> {
    enum InputIdx : size_t { QUERY = 0, KEY, VALUE, BLOCK_MASK, SCALE, COUNT };

    CLDNN_DECLARE_PRIMITIVE(block_sparse_attention)

    block_sparse_attention() : primitive_base("", {}) {}

    /// @brief Constructs block_sparse_attention primitive / layer.
    ///
    /// @param id             An identifier of new primitive.
    /// @param inputs         A list of input primitive ids: [query, key, value, block_mask] or
    ///                       [query, key, value, block_mask, scale].
    /// @param block_size_q   Query block size, > 0.
    /// @param block_size_kv  Key/value block size, > 0.
    /// @param causal         Token-level causal masking, ANDed with block_mask.
    block_sparse_attention(const primitive_id& id, const std::vector<input_info>& inputs, int64_t block_size_q, int64_t block_size_kv, bool causal = false)
        : primitive_base(id, inputs),
          block_size_q(block_size_q),
          block_size_kv(block_size_kv),
          causal(causal) {
        OPENVINO_ASSERT(inputs.size() == 4 || inputs.size() == 5, "[GPU] Unexpected input number for block_sparse_attention primitive: ", inputs.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, block_size_q);
        seed = hash_combine(seed, block_size_kv);
        seed = hash_combine(seed, causal);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const block_sparse_attention>(rhs);
        return block_size_q == rhs_casted.block_size_q && block_size_kv == rhs_casted.block_size_kv && causal == rhs_casted.causal;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<block_sparse_attention>::save(ob);
        ob << block_size_q;
        ob << block_size_kv;
        ob << causal;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<block_sparse_attention>::load(ib);
        ib >> block_size_q;
        ib >> block_size_kv;
        ib >> causal;
    }

    int64_t block_size_q = 0;
    int64_t block_size_kv = 0;
    bool causal = false;
};

// Backward-compatible alias for the (now nested) input-index enum, so existing call sites that
// spell `BlockSparseAttentionInputIdx::QUERY` keep working without qualification.
using BlockSparseAttentionInputIdx = block_sparse_attention::InputIdx;

}  // namespace cldnn
