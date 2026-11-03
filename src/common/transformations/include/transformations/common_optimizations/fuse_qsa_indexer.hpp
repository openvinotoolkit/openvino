// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses QSAPrepare and SparseAttentionIndexer into the stateful internal QSAIndexer op, leaving query-side
 * RMSNorm and RoPE outside the op (query_norm and query_rope disabled).
 *
 *        Query  RawKeys cos/sin ... KeyLength           Query
 *          │       │       │            │                 │
 *          │    ┌──┴───────┴────────────┴──┐          Transpose  RawKeys cos/sin ... KeyLength
 *          │    │        QSAPrepare        │              │      [layout adapters, cache growth]
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
 *  - the block-key cache is grown by zero blocks up to the complete blocks of the longest sequence, which
 *    is the shape of the QSAPrepare output, since QSAIndexer keeps the cache shape;
 *  - q_norm_weight is a vector of ones.
 * block_topk is token_budget / key_block_size and scale is the default 1 / sqrt(head size).
 *
 * The rewrite is applied only when the query and the raw keys have compatible sequence lengths, the query
 * head count and head size are static, RoPE and the key norm are enabled, both source ops use the same
 * key_block_size and key length, and SparseAttentionIndexer consumes QSAPrepare output 0, has no index weights
 * and selects with one head. The other requirements are validated by the source ops.
 */
class TRANSFORMATIONS_API FuseQSAIndexerBase : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexerBase");

    FuseQSAIndexerBase();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Moves the query RoPE into a QSAIndexer created by FuseQSAIndexerBase, enabling query_rope.
 *
 *   x ─► RoPE(input_trans0213) ─► Transpose(0, 2, 1, 3) ─► QSAIndexer   ──►   x ─► QSAIndexer(query_rope)
 *
 * The RoPE is the plain rotate-half RoPE of the QSAIndexer rotary dimension, with Unsqueeze(axis=1) of the
 * QSAIndexer cos / sin (before their batch Broadcast). It must precede the query norm in QSAIndexer, so it
 * is fused only while query_norm and query_rope are disabled.
 */
class TRANSFORMATIONS_API FuseQSAIndexerQueryRoPE : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexerQueryRoPE");

    FuseQSAIndexerQueryRoPE();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Moves the query RMSNorm into a QSAIndexer, enabling query_norm.
 *
 *   x ─► RMS ─► QSAIndexer   ──►   x ─► QSAIndexer(query_norm)
 *
 * The RMS is the non-affine RMS of the last axis that keeps the precision, and its epsilon equals the
 * QSAIndexer epsilon. It is fused only while query_norm is disabled.
 */
class TRANSFORMATIONS_API FuseQSAIndexerQueryNorm : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexerQueryNorm");

    FuseQSAIndexerQueryNorm();
};

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses QSAPrepare and SparseAttentionIndexer into QSAIndexer, then moves the query RoPE and the query
 * RMSNorm into it.
 *
 * If the model has SparseAttentionIndexer, RoPEFusion (with 2D RoPE support) and RMSFusion (without tail Convert,
 * with non-affine RMS) run model-wide first, so the query RoPE and RMSNorm are RoPE and RMS nodes. The nested
 * passes keep the caller's disabled passes and callbacks. In the GPU pipeline it runs before FuseRMSRoPE
 * combines RMS and RoPE.
 * The Qwen query (non-affine RMS on [B, L, H, D], then RoPE with input_trans0213) gives the GPU contract, a
 * QSAIndexer with query_norm and query_rope enabled.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("FuseQSAIndexer");

    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass
