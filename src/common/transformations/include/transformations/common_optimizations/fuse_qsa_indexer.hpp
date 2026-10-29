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
 * The rewrite is applied only when:
 *  - QSAIndexer::is_supported accepts the resulting configuration and query shape (kernel limits on
 *    head count, head size, key block size, selection heads, token budget and epsilons);
 *  - query has rank 4 and raw keys have rank 3, with compatible batch and sequence;
 *  - both source ops use the same key_block_size and the same key length input;
 *  - SparseAttentionIndexer consumes QSAPrepare output 0 and has no index weights input;
 *  - both RMSNorm weights are enabled (not empty Constants);
 *  - every RMSNorm node and every QSAPrepare output has a single consumer, since all of them are
 *    replaced by QSAIndexer.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexer");
    FuseQSAIndexer();
};

}  // namespace ov::pass
