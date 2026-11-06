// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

inline constexpr char grouped_matmul_rows_per_token_key[] = "intel_gpu_grouped_matmul_rows_per_token";

/**
 * @brief Marks a MoE GroupedMatMul with the number of its rows per token (top_k).
 *
 * The router top-k ids of one token are distinct, so after MoEExpertSort no expert gets more
 * rows than there are tokens: max group size <= rows / top_k. The GPU grouped GEMM passes
 * this bound to oneDNN as a dispatch hint.
 *
 *   router ─ TopK | MoERouterFused ─(ids [T, top_k])─ (Convert) ─ MoEExpertSort
 *                                                                     │ offsets
 *                                                                 (Convert)
 *                                                                     │
 *               mat_a [T*top_k, K], mat_b ───────────────────── GroupedMatMul(Compressed)
 *
 * Nothing is rewritten: the matched GroupedMatMul gets rt_info[grouped_matmul_rows_per_token_key] = top_k.
 * The ids must be the indices output of a TopK over the last axis of a 2D input with static k,
 * or of MoERouterFused.
 */
class MarkGroupedMatMulRowsPerToken : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkGroupedMatMulRowsPerToken");
    MarkGroupedMatMulRowsPerToken();
};

}  // namespace ov::intel_gpu
