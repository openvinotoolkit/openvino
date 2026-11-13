// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <type_traits>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::op::internal {
class QSAIndexer;
class PagedQSAIndexer;
}  // namespace ov::op::internal

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses QSAPrepare and SparseAttentionIndexer into the stateful internal QSAIndexer op, leaving query-side
 * RMSNorm and RoPE outside the op (query_norm and query_rope disabled).
 * The disabled flags are intermediate fusion states only: query preprocessing must be absorbed by
 * FuseQSAIndexerQuery and both flags must be true before the indexer is executed.
 *
 *        Query  RawKeys cos/sin ... KeyLength           Query
 *          │       │       │            │                 │
 *          │    ┌──┴───────┴────────────┴──┐          Transpose  RawKeys cos/sin ... KeyLength
 *          │    │        QSAPrepare        │              │      [layout adapters]
 *          │    └──┬────────┬──────────┬───┘           ┌──┴──────────┴───────┴───────────┴──┐
 *   ┌──────┴───────┴──┐     │          │      ──►      │             QSAIndexer             │
 *   │ SparseAttention │  incomplete keys / rope        └──┬──────┬───────┬───────┬──────────┘
 *   │     Indexer     │                               indices count  Squeeze Squeeze   incomplete
 *   └──┬──────────┬───┘                                              block   incomplete  rope
 *   indices    count                                                 keys    keys
 *
 * QSAIndexer takes [B, L, H, D] activations and head-first states, so the pass adds adapters:
 *  - the [B, H, L, D] query is transposed;
 *  - the raw keys get a head axis 2, and the block-key cache and incomplete keys get a head axis 1,
 *    which is squeezed from the matching QSAIndexer outputs;
 *  - key_length is converted to i32 and broadcast unless its statically known length equals the raw-key
 *    batch size;
 *  - cos / sin are broadcast unless their statically known batch sizes equal the raw-key batch size;
 *    dynamic batch dimensions are not assumed equal without shape-symbol propagation;
 *  - the block-key cache is passed as is, since QSAIndexer follows the QSAPrepare cache contract;
 *  - q_norm_weight is a vector of ones.
 * block_topk is token_budget / key_block_size and scale is set explicitly to the default 1 / sqrt(head size).
 *
 * The rewrite is applied only when the query and the raw keys have compatible sequence lengths, the query
 * head count and head size are static, RoPE and the key norm are enabled, both source ops use the same
 * key_block_size and key length, and SparseAttentionIndexer consumes QSAPrepare output 0, has no index weights
 * and selects with one head. The other requirements are validated by the source ops.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexer");

    FuseQSAIndexer();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Moves the query RoPE into a dense or paged QSA indexer, enabling query_rope.
 *
 *   x ─► RoPE(input_trans0213) ─► Transpose(0, 2, 1, 3) ─► QSAIndexer   ──►   x ─► QSAIndexer(query_rope)
 *
 * The RoPE is the plain rotate-half RoPE of the QSAIndexer rotary dimension, with Unsqueeze(axis=1) of the
 * QSAIndexer cos / sin (before their batch Broadcast). A Reshape with a constant target from [B, L, D] to
 * [B, 1, L, D] is also supported when L and D are static and positive. It is fused while query_rope is disabled.
 * QSAIndexer normalizes before it rotates, so it must run before FuseQSAIndexerQueryNorm, as FuseQSAIndexerQuery
 * does; the query_norm of the matched indexer is not checked.
 * The PagedQSAIndexer specialization requires the layout adapters produced by PagedQSAIndexerFusion:
 * query [B, L, H, D] -> [B * L, H * D] and cos/sin [B, L, rotary_dim] -> [B * L, rotary_dim].
 * It relies on that contract rather than revalidating the layouts. The patterns use optional rank-2
 * Reshape adapters, which the dense path bypasses. The query Reshape must feed only the indexer; it is rewired
 * to the unprocessed query.
 * Both specializations share PassConfig disable settings and callbacks, which receive the concrete indexer node.
 */
template <class Indexer>
class TRANSFORMATIONS_API FuseQSAIndexerQueryRoPE : public MatcherPass {
    static_assert(std::is_same_v<Indexer, ov::op::internal::QSAIndexer> ||
                      std::is_same_v<Indexer, ov::op::internal::PagedQSAIndexer>,
                  "FuseQSAIndexerQueryRoPE supports QSAIndexer and PagedQSAIndexer");

public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexerQueryRoPE");

    FuseQSAIndexerQueryRoPE();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Moves the query RMSNorm into a dense or paged QSA indexer, enabling query_norm.
 *
 *   x ─► RMS ─► QSAIndexer   ──►   x ─► QSAIndexer(query_norm)
 *
 * The RMS is the non-affine RMS of the last axis that keeps the precision, and its epsilon equals the
 * QSAIndexer epsilon. It is fused while query_norm is disabled, regardless of query_rope.
 * The PagedQSAIndexer specialization requires the layout adapters produced by PagedQSAIndexerFusion.
 * It matches RMS before the query's [B, L, H, D] -> [B * L, H * D] Reshape. The RMS axis
 * must be static and equal to indexer_head_dim, so a norm across flattened heads remains outside the indexer.
 * An optional rank-2 Reshape matches the paged layout adapter, which the dense path bypasses. It must feed only
 * the indexer and is rewired to the unprocessed query.
 * Both specializations share PassConfig disable settings and callbacks, which receive the concrete indexer node.
 */
template <class Indexer>
class TRANSFORMATIONS_API FuseQSAIndexerQueryNorm : public MatcherPass {
    static_assert(std::is_same_v<Indexer, ov::op::internal::QSAIndexer> ||
                      std::is_same_v<Indexer, ov::op::internal::PagedQSAIndexer>,
                  "FuseQSAIndexerQueryNorm supports QSAIndexer and PagedQSAIndexer");

public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexerQueryNorm");

    FuseQSAIndexerQueryNorm();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Absorbs canonical query RMS and RoPE into dense and paged QSA indexers.
 *
 * Register after RMSFusion and RoPEFusion, before passes that combine or decompose RMS/RoPE.
 * Does not run those model-wide canonicalization passes itself. Runs the dense and paged FuseQSAIndexerQueryRoPE
 * over the whole model first and FuseQSAIndexerQueryNorm second, with the caller's PassConfig, since QSAIndexer
 * normalizes before it rotates. Specializations share PassConfig callbacks and disable settings.
 * Paged indexers must retain the layout adapters generated by PagedQSAIndexerFusion.
 */
class TRANSFORMATIONS_API FuseQSAIndexerQuery : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseQSAIndexerQuery");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass
