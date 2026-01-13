// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/matcher_pass.hpp"
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
 *      ┌───────────────────────┬────────────┐
 *      ▼                       ▼            ▼
 * Packed MatMul_Q  Packed MatMul_K  Packed MatMul_V
 *      │                │                │
 *     Add_Q           Add_K            Add_V
 *      │                │                │
 *   ROPE (Q)         ROPE (K)          ROPE(V)
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

/**
 * @brief Fuses two unrolled Scaled Dot-Product Attention (SDPA) branches that are combined with an Add into a single,
 *        more compact SDPA representation.
 *
 * @details
 * This transformation is implemented as an ov::pass::MatcherPass and uses pattern-based matching to detect a subgraph
 * where SDPA has been "unrolled" into primitive operations in two parallel branches whose results are summed (Add).
 * When the pattern is found and it is safe to do so, the pass replaces the matched subgraph with an equivalent,
 * optimized form (typically reducing the number of operations and improving execution efficiency).
 *
 * The pass is intended for common-optimizations pipelines to simplify graphs produced by frontends or earlier
 * transformations, and can enable subsequent fusions and backend-specific optimizations.
 *
 * @note The pass should preserve graph semantics. It may decline to transform if required constraints (e.g. compatible
 *       shapes, attributes, constants, or single-consumer requirements) are not satisfied.
 */
class TRANSFORMATIONS_API MergeTwoUnrolledSDPAAdd : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeTwoUnrolledSDPAAdd");
    MergeTwoUnrolledSDPAAdd();
};
/**
 * ## Pseudo-graph example (before/after)
 *
 * The pass targets graphs where two equivalent SDPA branches are executed in parallel and then merged by Add.
 * A simplified (illustrative) topology looks like:
 *
 * ### Before (two unrolled SDPA branches + Add)
 *
 *   Q ──► PreQ1 ─┐
 *               ├──► SDPA_1 ─► Post_1 ─┐
 *   K ──► PreK1 ─┘                    │
 *   V ──► PreV1 ──────────────────────┘
 *
 *   Q ──► PreQ2 ─┐
 *               ├──► SDPA_2 ─► Post_2 ─┐
 *   K ──► PreK2 ─┘                    │
 *   V ──► PreV2 ──────────────────────┘
 *
 *                     Post_1 ─┐
 *                             ├──► Add ─► Output
 *                     Post_2 ─┘
 *
 * Where:
 * - Pre* may include: MatMul (+ Add bias), Dequantize, Reshape/Transpose, RoPE/positional ops, etc.
 * - Post_* may include: per-branch output projection, reshapes/transposes, etc.
 *
 * ### After (single packed/merged SDPA)
 *
 *   Q ──► Pack/PreQ ─┐
 *                   ├──► SDPA_Packed ─► Post_Packed ─► Output
 *   K ──► Pack/PreK ─┤
 *   V ──► Pack/PreV ─┘
 *
 * Notes:
 * - Exact packing/unpacking ops are implementation-specific (Concat/Reshape/Transpose/etc.).
 * - The pass only applies when it can prove equivalence/compatibility (e.g., same masks/scales, aligned shapes).
 */

class TRANSFORMATIONS_API MergeTwoUnrolledRoPEConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeTwoUnrolledRoPEConcat");
    MergeTwoUnrolledRoPEConcat();
};

class TRANSFORMATIONS_API MergeMatMulBiasConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeMatMulBiasConcat");
    MergeMatMulBiasConcat();
};

class TRANSFORMATIONS_API MergeKVCaches : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeKVCaches");
    MergeKVCaches();
};

class TRANSFORMATIONS_API MergeDQConcat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeDQConcat");
    MergeDQConcat();
};

class TRANSFORMATIONS_API PackMultiHeadAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PackMultiHeadAttention");
    PackMultiHeadAttention() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace ov::pass