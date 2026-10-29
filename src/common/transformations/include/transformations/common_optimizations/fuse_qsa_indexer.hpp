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
 *  - query is [?, 4, ?, 128] and raw keys are [?, ?, 128], with compatible batch and sequence;
 *  - key_block_size is 4, num_selection_heads is 1 and token_budget <= 512;
 *  - both RMSNorm epsilons are in [1e-6, 1e-5] and both norm weights are enabled;
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
