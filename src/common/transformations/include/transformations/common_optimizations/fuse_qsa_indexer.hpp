// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses query RMSNorm and RoPE, QSAPrepare and SparseAttentionIndexer into the internal QSAIndexer op.
 * QSAIndexer is backend-neutral; execution requires a device implementation. The GPU implementation supports
 * f16 inputs and a single index-key/selection head.
 *
 *   raw query -> RMS -> RoPE(input_trans0213) --------+
 *   raw keys, cos/sin, states -> QSAPrepare ---------+-> SparseAttentionIndexer
 *
 *                              |
 *                              v
 *
 *   raw query, raw keys, cos/sin, weights, adapted states -> QSAIndexer -> selections, updated states
 *
 * QSAIndexer takes [B, L, H, D] activations and head-first states, so the pass adds adapters:
 *  - the raw [B, L, H, D] query bypasses its RMSNorm and RoPE, which the indexer applies internally;
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
 *
 * Both query steps must match: non-affine, precision-preserving RMSNorm of the last axis with exactly the
 * QSAPrepare epsilon, followed by plain rotate-half RoPE with input_trans0213 and the same cos/sin rows as
 * QSAPrepare. RoPE rows have an Unsqueeze(axis=1), or a constant-target Reshape from [tokens, 1, D] to
 * [tokens, 1, 1, D] with static positive D. Partial query fusion is not supported.
 * Register after the caller's RMSFusion and RoPEFusion; this pass does not run model-wide canonicalization.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexer");

    FuseQSAIndexer();
};

}  // namespace ov::pass
