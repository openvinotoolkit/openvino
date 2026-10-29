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
 * The rewrite is applied only when the matched graph is within the QSAIndexer kernel limits
 * (QSAIndexer::supported_* / max_token_budget / [min_epsilon, max_epsilon]):
 *  - query is [?, supported_num_heads, ?, supported_head_size] and raw keys have rank 3,
 *    with compatible batch and sequence;
 *  - both source ops use supported_key_block_size, SparseAttentionIndexer uses
 *    supported_num_selection_heads and a token_budget in (0, max_token_budget] divisible by the
 *    block size;
 *  - both RMSNorm epsilons are in [min_epsilon, max_epsilon] and both norm weights are enabled;
 *  - SparseAttentionIndexer consumes QSAPrepare output 0 and the same key length as QSAPrepare,
 *    and has no index weights input;
 *  - every RMSNorm node and every QSAPrepare output has a single consumer, since all of them are
 *    replaced by QSAIndexer.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexer");
    FuseQSAIndexer();
};

}  // namespace ov::pass
