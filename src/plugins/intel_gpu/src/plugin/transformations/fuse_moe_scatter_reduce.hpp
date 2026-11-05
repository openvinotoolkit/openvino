// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Fuses the combine tail of a GroupedMatMul MoE block into MoEScatterReduce.
 *
 * The generic grouped experts forward (transformers `grouped_mm_experts_forward`,
 * optimum-intel `ov_grouped_mm`) restores token order with the inverse permutation, weights the
 * rows by the router weights and sums the top-k rows of each token:
 *
 *   rows     = Gather(down_out [S, H], row_lut [S], axis 0)
 *   weighted = Multiply(rows, Unsqueeze(Reshape(weights [T, K], [-1]), -1))
 *   out      = ReduceSum(Reshape(weighted, [-1, K, H]), axis 1)
 *
 * Before:                                              After:
 *
 *   down_out   row_lut    weights                        down_out  weights  row_lut
 *       └─ Gather ─┘         │                                └───────┼───────┘
 *            └── Multiply ── Unsqueeze ─ Reshape               ┌───────┴────────┐
 *                   │                                          │MoEScatterReduce│
 *                Reshape ─ ReduceSum                           └───────┬────────┘
 *
 * row_lut must be the third output of MoEExpertSort (FuseMoEExpertSort runs first), optionally
 * through Converts.
 */
class FuseMoEScatterReduce : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoEScatterReduce");
    FuseMoEScatterReduce();
};

}  // namespace ov::intel_gpu
