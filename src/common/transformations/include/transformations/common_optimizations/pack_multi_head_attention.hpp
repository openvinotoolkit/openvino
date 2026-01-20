// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <utility>

#include "openvino/pass/matcher_pass.hpp"
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
 * - Replaces Add chains for attention output merging with a ReduceSum
 *
 * ## Before (unrolled heads), for illustration purpose (see the patterns for details)
 *
 *             ┌──────────┐
 *             │  Input   │
 *             └────┬─────┘
 *                  │
 *     ┌────────────┼───────────┬────────────┐
 *     ▼            ▼           ▼            ▼
 *   MatMul_Q1    MatMul_Q2    ...        MatMul_QN
 *   MatMul_K1    MatMul_K2    ...        MatMul_KN
 *   MatMul_V1    MatMul_V2    ...        MatMul_VN
 *     │            │                        │
 *    Add_Q1      Add_Q2       ...         Add_QN
 *    Add_K1      Add_K2       ...         Add_KN
 *    Add_V1      Add_V2       ...         Add_VN
 *     │            │                        │
 *   ROPE_Q1     ROPE_Q2       ...           |
 *   ROPE_K1     ROPE_K2       ...           |
 *     │            │                        │
 *   SDPA_1       SDPA_2       ...         SDPA_N
 *     │            │                        │
 *  Linear_1     Linear_2      ...        Linear_N
 *     │            │                        │
 *     |            │                        │
 *     └── Add_1 ───|                        |
 *                  |                        |
 *                  └─ Add_2 ─ ...           |
 *                       |                   |
 *                       │                   |
 *                       └──── Add_N ────────┘
 *                                |
 *                              Output
 *
 * ## After (packed/fused heads)
 *
 *        ┌──────────────┐
 *        │    Input     │
 *        └──────┬───────┘
 *               │
 *      ┌───────────────┬─────────────────┐
 *      ▼               ▼                 ▼
 * Packed MatMul_Q  Packed MatMul_K  Packed MatMul_V
 *      │                │                │
 *     Add_Q           Add_K            Add_V
 *      │                │                │
 *   ROPE (Q)         ROPE (K)            |
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

/// \brief Common: matcher-based transformations used by PackMultiHeadAttention.
/// \details These passes merge/pack parts of an unrolled Multi-Head Attention (MHA) / Grouped Query Attention (GQA)
///          subgraph step-by-step to keep patterns simpler and the transformation flow clearer.

/// \brief Description: merges unrolled Scaled Dot-Product Attention (SDPA) subgraphs.
/// \details Detects multiple per-head SDPA branches and replaces them with a single SDPA operating on packed Q/K/V.
class TRANSFORMATIONS_API MergeUnrolledSDPA : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeUnrolledSDPA");
    MergeUnrolledSDPA();
};

/// \brief Description: merges unrolled Rotary Positional Embedding (RoPE) subgraphs.
/// \details Detects per-head RoPE (or similar positional embedding) applied to Q/K and converts it to a packed form.
class TRANSFORMATIONS_API MergeUnrolledRoPE : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeUnrolledRoPE");
    MergeUnrolledRoPE();
};

/// \brief Description: merges linear projection subgraphs (e.g., Q/K/V projections).
/// \details Fuses multiple per-head projection MatMuls (optionally with Add and/or dequantization) into packed MatMuls.
class TRANSFORMATIONS_API MergeLinearProjections : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeLinearProjections");
    MergeLinearProjections();
};

/// \brief Description: concatenates K/V caches.
/// \details Concatenates KV-cache read/write subgraphs used by attention.
class TRANSFORMATIONS_API MergeKVCaches : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeKVCaches");
    MergeKVCaches();
};

/// \brief Description: merges Dequantize (DQ) subgraphs used in MHA.
/// \details Detects repeated per-head dequantization patterns and merges them to enable packing of projections.
class TRANSFORMATIONS_API MergeDQ : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MergeDQ");
    MergeDQ();
};

/// \brief Common: model-level transformation that orchestrates packing/canonicalization of MHA/GQA.
/// \details Runs step-by-step merging passes (projections, RoPE, SDPA, KV-cache, DQ) to produce a compact packed form.
class TRANSFORMATIONS_API PackMultiHeadAttention : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("PackMultiHeadAttention");
    PackMultiHeadAttention() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

class PackMultiHeadAttentionRTInfo : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("PackMultiHeadAttentionRTInfo", "0", ov::RuntimeAttribute);
    PackMultiHeadAttentionRTInfo() = default;
    bool is_copyable() const override {
        return false;
    }
};

}  // namespace ov::pass