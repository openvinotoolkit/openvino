// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include "openvino/core/node_vector.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/pass/pattern/op/block.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass::pattern::op {

class TRANSFORMATIONS_API QSAIndexQueryBlock;

}  // namespace ov::pass::pattern::op

/**
 * @brief The index query of a Qwen Sparse Attention indexer as regular graph operations: the RMSNorm and the RoPE that
 * a fused indexer (CPU / GPU QSAIndexer, PagedQSAIndexer) applies itself to the raw query.
 *
 * The block output is the `index_query` input of SparseAttentionIndexer, [N, H, L, D]. Accepted forms, with optional
 * Converts between the operations:
 *
 *     raw -> [Transpose] -> RMS(x, gain) | RMS(x) [* gain] -> [Transpose] -> RoPE                  -> [Transpose]
 *     raw -> [Transpose] -> RMS(x, gain) | RMS(x) [* gain] -> [Transpose] -> Concat(RoPE(x[..., 0:Dr]),
 *                                                                                x[..., Dr:D])    -> [Transpose]
 *
 * - RMS(x) without a gain is an RMSNorm whose gain is all ones (e.g. a zero `w` of a `1 + w` norm, folded away).
 * - The RoPE is a plain rotate-half one on Dr <= D dimensions; it may absorb the [N, L, H, D] -> [N, H, L, D] transpose
 *   of its input (input_trans0213). A partial rotation is either a single RoPE or the Concat form, with Slice,
 *   StridedSlice or VariadicSplit parts, depending on the RoPE fusions of the pipeline.
 * - The transposes swap the axes 1 and 2, so the raw query is [N, L, H, D] or [N, H, L, D] depending on their count.
 */
class ov::pass::pattern::op::QSAIndexQueryBlock : public ov::pass::pattern::op::Block {
public:
    QSAIndexQueryBlock();

    /// A SparseAttentionIndexer fed by QSAPrepare and by this block, as a fused indexer takes it.
    struct Match {
        std::shared_ptr<ov::op::v17::SparseAttentionIndexer> indexer;
        std::shared_ptr<ov::op::v17::QSAPrepare> prepare;
        ov::Output<ov::Node> raw_query;      ///< the RMSNorm input
        bool raw_query_blhd = false;         ///< raw_query is [N, L, H, D]; [N, H, L, D] otherwise
        ov::Output<ov::Node> q_norm_weight;  ///< the RMSNorm gain, D elements; empty when the gain is all ones
        size_t num_heads = 0;                ///< H, index query heads
        size_t head_dim = 0;                 ///< D
        size_t rotary_dim = 0;               ///< Dr
        size_t key_block_size = 0;           ///< r
        size_t block_topk = 0;               ///< token_budget / r
        float epsilon = 0.0f;                ///< of QSAPrepare, shared by the query RMSNorm
        ov::NodeVector nodes;                ///< the matched operations of the block, for the runtime info
    };

    /// Checks what the pattern cannot express: `indexer` (matched with this block as its index query) has to be fed by
    /// QSAPrepare with the same key lengths, both have to use the same RoPE tables, RMSNorm epsilon and key block size,
    /// and the attributes have to be expressible by a fused indexer. Returns std::nullopt otherwise.
    std::optional<Match> get_match(const PatternValueMap& pattern_map,
                                   const std::shared_ptr<ov::op::v17::SparseAttentionIndexer>& indexer) const;
};
