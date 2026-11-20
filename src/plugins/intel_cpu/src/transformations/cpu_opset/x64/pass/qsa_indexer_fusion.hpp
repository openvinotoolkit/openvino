// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/// \brief Fuses the Qwen Sparse Attention (QSA) index selection subgraph into one QSAIndexer op:
///
///   index_query = RoPE(RMS(q_raw, q_norm_weight), rope_cos, rope_sin)   (regular graph ops, [N, Hi, L, Di])
///   {block_keys, tail_keys, tail_rope} = QSAPrepare-17(k_raw, ..., rope_cos, rope_sin, k_norm_weight, ...)
///   {sel_indices, sel_count} = SparseAttentionIndexer-17(index_query, block_keys, key_length)
///
/// The query RMSNorm and RoPE are matched as the internal RMS / RoPE ops, so the pass has to run after
/// RMSFusion and RoPEFusion. The pass only fuses what QSAIndexer can express: one index key head, one selection
/// head, unit index weights, enabled RoPE and key normalization, and the query RoPE tables being the very
/// tables QSAPrepare rotates the block keys with.
class QSAIndexerFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("QSAIndexerFusion");
    QSAIndexerFusion();
};

}  // namespace ov::intel_cpu
