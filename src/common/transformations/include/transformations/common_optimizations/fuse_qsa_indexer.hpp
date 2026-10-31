// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {

/**
 * @ingroup ov_transformation_common_api
 * @brief Device-specific QSAIndexer kernel limits used by FuseQSAIndexer.
 *
 * The default configuration adds no limits of its own, and the CPU and GPU pipelines use it;
 * a plugin may pass a narrower configuration that matches its QSAIndexer kernels.
 */
struct FuseQSAIndexerConfig {
    /// Allowed SparseAttentionIndexer num_selection_heads values; empty allows any value.
    std::vector<int64_t> supported_num_selection_heads;
    /// When set, the query and key RMSNorm epsilons must agree:
    /// |q_eps - k_eps| <= max_epsilon_relative_difference * k_eps. Used by kernels with a single epsilon.
    std::optional<float> max_epsilon_relative_difference;
};

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
 *  - query has QSAIndexer::query_rank and raw keys have QSAIndexer::keys_rank, with compatible
 *    batch and sequence; the block-key cache and incomplete keys have QSAIndexer::keys_rank;
 *  - the query head count, the head size and the rotary dimension are static, since QSAIndexer
 *    kernels are specialized for them;
 *  - RoPE is enabled, since QSAIndexer kernels always apply it: rope_cos / rope_sin have
 *    QSAIndexer::rope_rank and the incomplete RoPE state has QSAIndexer::incomplete_rope_rank;
 *  - both source ops use the same key_block_size and the same key length input;
 *  - both RMSNorm weights are enabled;
 *  - SparseAttentionIndexer consumes QSAPrepare output 0 and has no index weights input;
 *  - every RMSNorm node and every QSAPrepare output has a single consumer, since all of them are
 *    replaced by QSAIndexer;
 *  - the graph is within the device-specific kernel limits of FuseQSAIndexerConfig: selection
 *    head count and query/key epsilon agreement.
 *
 * Head count, head size, key block size and token budget values are taken from the graph and are
 * not fixed, and element types are not restricted. Token budget positivity and divisibility, the
 * minimum key block size, and the rotary dimension bounds are guaranteed by SparseAttentionIndexer
 * and QSAPrepare validation.
 */
class TRANSFORMATIONS_API FuseQSAIndexer : public MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseQSAIndexer");
    using Config = FuseQSAIndexerConfig;

    FuseQSAIndexer();
    explicit FuseQSAIndexer(const Config& config);
};

}  // namespace ov::pass
