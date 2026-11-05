// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Fuses the expert sort of a GroupedMatMul MoE block into MoEExpertSort.
 *
 * The generic grouped experts forward (transformers `grouped_mm_experts_forward`,
 * optimum-intel `ov_grouped_mm`) sorts the flattened top-k expert ids with two argsorts and
 * counts the rows of each expert with a scatter-add:
 *
 *   flat_ids = Reshape(topk_ids [T, K], [-1])                        [S = T*K]
 *   perm     = TopK(flat_ids, k = S, min).indices                    argsort
 *   offsets  = CumSum(ScatterElementsUpdate(zeros[E], flat_ids[perm], ones, sum))
 *   token    = Reshape(Broadcast(Unsqueeze(Range(0, T, 1)), [T, K]), [-1])[perm]
 *   inv_perm = TopK(perm, k = S, min).indices                        argsort of the permutation
 *
 * Before:                                            After:
 *
 *   topk_ids ─ Reshape ─ TopK ─┬─ Gather ─ ScatterElementsUpdate ─ CumSum      topk_ids
 *                              ├─ Gather(token)                                  │
 *                              └─ TopK ─ (Convert)                         ┌─────┴───────┐
 *                                                                          │MoEExpertSort│
 *                                                                          └─┬────┬────┬─┘
 *                                                                        token offsets row_lut
 *
 * The expert sort of MoEExpertSort is stable by flat position, while the argsort TopK is not.
 * Rows of one expert are independent, so the result does not change as long as every user of the
 * permutation is replaced. The rewrite is therefore applied only when the permutation feeds nothing
 * but the counted ids, the token index, the inverse argsort and shape computations.
 * Optional Converts are allowed on the permutation, the sorted ids and the inverse permutation.
 * The number of experts must not exceed max_num_experts (a single work-group of the GPU kernel).
 */
class FuseMoEExpertSort : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoEExpertSort");
    explicit FuseMoEExpertSort(size_t max_num_experts);
};

}  // namespace ov::intel_gpu
