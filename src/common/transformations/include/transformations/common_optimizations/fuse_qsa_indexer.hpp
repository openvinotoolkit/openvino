// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Fuses QSAPrepare and SparseAttentionIndexer into the stateful internal QSAIndexer op.
 *
 * The query RMSNorm, key block preparation and block scoring are executed by one kernel, which
 * avoids materializing the intermediate block index keys and updated cache state separately.
 *
 * Before:                                            After:
 *
 *   Query  QWeight   RawKeys ... KeyLength ...         Query QWeight RawKeys ... KeyLength ...
 *     │       │         │            │                    │     │       │            │
 *   ┌─┴───────┴─┐   ┌───┴────────────┴───┐             ┌──┴─────┴───────┴────────────┴──┐
 *   │  RMSNorm  │   │     QSAPrepare     │             │            QSAIndexer          │
 *   └─────┬─────┘   └──┬──────┬──────┬───┘             └─┬────┬────────┬────────┬────┬──┘
 *         │   BlockKeys│      │      │                 indices count  block  incomplete
 *       ┌─┴────────────┴──────┴─┐    │                                 keys   keys/rope
 *       │SparseAttentionIndexer │    │ incomplete keys/rope state
 *       └───┬──────────────┬────┘    │
 *        indices         count
 *
 * RMSNorm is either the internal RMSNorm op or its decomposition
 * Multiply(Divide(x, Sqrt(Add(ReduceMean(Power(x, 2), last_axis, keep_dims), eps))), weight).
 *
 * The rewrite is applied only when the matched graph fits the QSAIndexer kernels. The numeric
 * limits are defined once as QSAIndexer constants (aligned with the GPU QSAIndexerFusion limits):
 *  - query has QSAIndexer::query_rank with static head count and head size, and raw keys have
 *    QSAIndexer::keys_rank with the same static head size; batch and sequence of query and raw
 *    keys are compatible;
 *  - RoPE is enabled: rope_cos / rope_sin have QSAIndexer::rope_rank with a static rotary
 *    dimension and the incomplete RoPE state has QSAIndexer::incomplete_rope_rank; the block-key
 *    cache and incomplete keys have QSAIndexer::keys_rank;
 *  - both source ops use the same key_block_size and the same key length input, and
 *    SparseAttentionIndexer uses QSAIndexer::supported_num_selection_heads;
 *  - both RMSNorm weights are enabled and the query and key epsilons agree:
 *    |q_eps - k_eps| <= QSAIndexer::max_epsilon_relative_difference * k_eps;
 *  - SparseAttentionIndexer consumes QSAPrepare output 0 and has no index weights input;
 *  - every RMSNorm node and every QSAPrepare output has a single consumer, since all of them are
 *    replaced by QSAIndexer.
 *
 * Token budget positivity and divisibility, the minimum key block size, and the rotary dimension
 * bounds are guaranteed by SparseAttentionIndexer and QSAPrepare validation.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexer");
    FuseQSAIndexer();
};

}  // namespace ov::pass
