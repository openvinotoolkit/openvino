// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/pattern/multi_matcher.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {
/**
 * @brief Detects and fuses unrolled MultiHeadAttention (MHA) or Grouped Query Attention (GQA) subgraphs.
 *
 * This transformation identifies subgraphs where Q, K, V projections and attention heads
 * are computed independently and merges them into a packed multi-head format.
 *
 * ## What it does
 * - Detects multiple per-head Q/K/V branches: (MatMul [+Add], possibly with dequantization)
 * - Identifies rotary/positional embedding (ROPE) or similar pre-processing for Q/K
 * - Detects multiple SDPA (Scaled Dot-Product Attention) blocks (per head)
 * - Fuses all Q MatMuls into a single packed MatMul for Q; does the same for K and V
 * - Replaces N SDPA branches with a single SDPA operating on packed Q/K/V
 * - Fuses post-attention per-head output projections (MatMul/Add) into one
 * - Replaces Add/Concat chains for attention output merging with a ReduceSum
 *
 * ## Before (unrolled heads), for illustration purpose (see the patterns for details)
 *
 *             ┌──────────┐
 *             │  Input   │
 *             └────┬─────┘
 *                  │
 *     ┌────────────┼────────────┬─────────────┐
 *     ▼            ▼            ▼             ▼
 *   MatMul_Q1    MatMul_Q2   ...         MatMul_QN
 *   MatMul_K1    MatMul_K2   ...         MatMul_KN
 *   MatMul_V1    MatMul_V2   ...         MatMul_VN
 *     │            │                        │
 *    Add_Q1      Add_Q2      ...         Add_QN
 *    Add_K1      Add_K2      ...         Add_KN
 *    Add_V1      Add_V2      ...         Add_VN
 *     │            │                        │
 *   ROPE_Q1     ROPE_Q2      ...         ROPE_QN
 *   ROPE_K1     ROPE_K2      ...         ROPE_KN
 *     │            │                        │
 *   SDPA_1       SDPA_2      ...         SDPA_N
 *     │            │                        │
 *  Linear_1     Linear_2     ...        Linear_N
 *     │            │                        │
 *     └────────────┴───────── ... ──────────┘
 *                 │   (Add or Concat)
 *               Output
 *
 * ## After (packed/fused heads)
 *
 *        ┌──────────────┐
 *        │    Input     │
 *        └──────┬───────┘
 *               │
 *    ┌───────────────────────┬────────────┐
 *    ▼                       ▼            ▼
 * Packed MatMul_Q  Packed MatMul_K  Packed MatMul_V
 *      │                │                │
 *     Add_Q           Add_K            Add_V
 *      │                │                │
 *   ROPE (Q)         ROPE (K)           │
 *      │                │                │
 *      └─────────┬──────┴───────┬────────┘
 *                ▼              ▼
 *          (Packed Q)      (Packed K)
 *                   │        │
 *                ┌──▼────────▼──┐
 *                │   SDPA       │  (Single SDPA for all heads)
 *                └─────┬────────┘
 *                      │
 *             Linear Projection (packed)
 *                      │
 *                 ReduceSum
 *                      │
 *                   Output
 *
 * @ingroup ov_transformation_common_optimizations
 */

}  // namespace ov::pass

namespace ov::pass {

class TRANSFORMATIONS_API PackGQA : public ov::pass::MultiMatcher {
public:
    OPENVINO_RTTI("PackGQA");

    PackGQA();
};

}  // namespace ov::pass