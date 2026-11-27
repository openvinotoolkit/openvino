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
 * Before:
 *
 *               topk_ids [T, K]
 *                     │
 *                ┌────┴────┐
 *                │ Reshape │
 *                └────┬────┘
 *                     │ flat_ids [S]
 *                ┌────┴────┐
 *                │  TopK   │ argsort
 *                └────┬────┘
 *                     │ perm [S]
 *        ┌────────────┼┄┄┄┄┄┄┄┄┄┄┄┄┄┐
 *        │            ┆             ┆
 *   ┌────┴────┐  ┌────┴────┐  ┌─────┴─────┐
 *   │ Gather  │  │ Gather  │  │   TopK    │
 *   │flat_ids │  │ token   │  │ (inverse) │
 *   └────┬────┘  │ source  │  └─────┬─────┘
 *        │       └────┬────┘        ┆
 *   ┌────┴────┐       ┆             ┆
 *   │ScatterEU│       ┆             ┆
 *   └────┬────┘       ┆             ┆
 *   ┌────┴────┐       ┆             ┆
 *   │ CumSum  │       ┆             ┆
 *   └────┬────┘       ┆             ┆
 *     offsets    token index    inv_perm
 *
 * After:
 *
 *               topk_ids [T, K]
 *                     │
 *             ┌───────┴───────┐
 *             │ MoEExpertSort │
 *             └───┬───┬───┬───┘
 *                 ┆   │   ┆
 *       token index   │   inv_perm (row_lut)
 *                  offsets
 *
 * ┆ - optional branches.
 *
 * MoEExpertSort is stable, while the argsort TopK is not, so the pass requires every user of the permutation to be replaced.
 * The token index and the inverse permutation are optional. max_num_experts is the number of experts the kernel supports.
 * Requires SymbolicPropagation to run before, to check that the token source has the number of tokens of topk_ids.
 */
class FuseMoEExpertSort : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMoEExpertSort");
    explicit FuseMoEExpertSort(size_t max_num_experts);
};

}  // namespace ov::intel_gpu
