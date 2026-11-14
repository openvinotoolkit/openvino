// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/// This pass transforms the following sub-graph to a single Scaled Dot Product Attention operation.
/// Before:
///     ┌───────┐     ┌───────┐    ┌───────┐
///     │   Q   │     │   K   │    │   V   │
///     └───┬───┘     └───┬───┘    └───┬───┘
///         │             │            │
///         │             │            │
///     ┌───┴───┐   ┌─────┴──────┐     │
///     │ MatMul│<──│ Transpose  │     │
///     └───┬───┘   | (Optional) │     │
///         │       └────────────┘     │
///     ┌───┴───┐    ┌─────────────┐   │
///     │  Add  │<───│AttentionMask│   │
///     └───┬───┘    | (Optional)  │   │
///         │        └─────────────┘   │
///     ┌───┴───┐                      │
///     │Softmax│                      │
///     └───┬───┘                      │
///         │                          │
///     ┌───┴───┐                      │
///     │ MatMul│<─────────────────────┘
///     └───┬───┘
///     ┌───┴───┐
///     │ Output│
///     └───────┘
///
/// After:
///     ┌───────┐    ┌───────┐    ┌───────┐    ┌─────────────┐
///     │   Q   │    │   K   │    │   V   │    │AttentionMask│
///     └───┬───┘    └───┬───┘    └───┬───┘    └──────┬──────┘
///         │            │            │               │
///         │            │            │               │
///     ┌───┴────────────┴────────────┴───────────────┴─┐
///     │           ScaledDotProductAttention           │
///     └────────────────────┬──────────────────────────┘
///                          │
///                          │
///                     ┌────┴────┐
///                     │  Output │
///                     └─────────┘
///
/// A where-style mask Select(cond, scores, fill) is fused as the additive mask Select(cond, 0, fill) when
/// fill is -inf or the lowest value of the score type. Exact for -inf and lowest f32/bf16. For lowest f16,
/// fully masked rows differ. A masked-out score that is +inf/NaN is not handled.
class TRANSFORMATIONS_API SDPAFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAFusionMatcher", "0");
    SDPAFusionMatcher();
};

class TRANSFORMATIONS_API SDPAReshapeFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAReshapeFusion", "0");
    SDPAReshapeFusion();
};

/// This pass transforms the following sub-graph with sinks to a single Scaled Dot Product Attention operation.
/// Before:
/// ┌───────┐     ┌───────┐     ┌───────┐    ┌───────┐
/// │ Sinks │     │   Q   │     │   K   │    │   V   │
/// └───┬───┘     └───┬───┘     └───┬───┘    └───┬───┘
///     │             │             │            │
///     │             │             │            │
///     │         ┌───┴───┐   ┌─────┴──────┐     │
///     │         │ MatMul│<──│ Transpose  │     │
///     │         └───┬───┘   | (Optional) │     │
///     │             │       └────────────┘     │
///     │         ┌───┴───┐    ┌─────────────┐   │
///     │         │  Add  │<───│AttentionMask│   │
///     │         └───┬───┘    | (Optional)  │   │
///     │             │        └─────────────┘   │
///     │     ┌───────┴────────┐                 │
///     │     │Multiply (scale)│                 │
///     │     └───────┬────────┘                 │
///     │             │                          │
///     │         ┌───┴───┐                      │
///     └────────>│Concat │                      │
///               └───┬───┘                      │
///                   │                          │
///               ┌───┴───┐                      │
///               │Softmax│                      │
///               └───┬───┘                      │
///                   │                          │
///             ┌─────┴──────┐                   │
///             │StridedSlice│                   │
///             └─────┬──────┘                   │
///                   │                          │
///               ┌───┴───┐                      │
///               │ MatMul│<─────────────────────┘
///               └───┬───┘
///               ┌───┴───┐
///               │ Output│
///               └───────┘
///
/// After:
///     ┌───────┐    ┌───────┐    ┌───────┐    ┌─────────────┐    ┌─────┐  ┌─────┐
///     │   Q   │    │   K   │    │   V   │    │AttentionMask│    │Sinks│  │Scale│
///     └───┬───┘    └───┬───┘    └───┬───┘    └──────┬──────┘    └──┬──┘  └──┬──┘
///         │            │            │               │              │        │
///         │            │            │               │              │        │
///     ┌───┴────────────┴────────────┴───────────────┴──────────────┴─┐      │
///     │                    ScaledDotProductAttention                 │──────┘
///     └────────────────────────────────┬─────────────────────────────┘
///                                      │
///                                      │
///                                 ┌────┴────┐
///                                 │  Output │
///                                 └─────────┘

class TRANSFORMATIONS_API SDPAFusionMatcherSinks : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAFusionMatcherSinks", "0");
    SDPAFusionMatcherSinks();
};

/// This pass fuses a "split-attention" sub-graph into a single ScaledDotProductAttention.
/// The pattern is produced by some Gemma models which compute attention against past KV
/// and current KV separately and merge via Concat/VariadicSplit.
///
/// Before:
///       ┌─────────┐     ┌─────┐     ┌───────┐
///       │ K_cache │     │  Q  │     │ K_new │
///       └────┬────┘     └──┬──┘     └───┬───┘
///            │          ┌──┴──┐         │
///            │       ┌──┘     └──┐      │
///            │       │           │      │
///       ┌────┴───────┴─┐    ┌────┴──────┴┐
///       │    MatMul    │    │   MatMul   │
///       └──────┬───────┘    └──────┬─────┘
///              │ qk_cache          │ qk_new
///              └──────────┐   ┌────┘
///                         │   │
///                     ┌───┴───┴───┐
///                     │  Concat   │ (axis=-1)
///                     └─────┬─────┘
///                           │            ┌─────────────┐
///                        ┌──┴──┐         │AttentionMask│
///                        │ Add │<────────┤             │
///                        └──┬──┘         └─────────────┘
///                           │
///                      ┌────┴────┐
///                      │ Softmax │ (axis=-1)
///                      └────┬────┘
///                           │
///                  ┌────────┴────────┐
///                  │  VariadicSplit  │ (axis=-1)
///                  └───┬─────────┬───┘
///                   pc │         │ pn
///       ┌─────────┐    │         │    ┌───────┐
///       │ V_cache │    │         │    │ V_new │
///       └────┬────┘    │         │    └───┬───┘
///            │      ┌──┘         └──┐     │
///            │      │               │     │
///       ┌────┴──────┴┐          ┌───┴─────┴──┐
///       │   MatMul   │          │   MatMul   │
///       └──────┬─────┘          └──────┬─────┘
///              │ attn_c                │ attn_n
///              └──────────┐   ┌────────┘
///                         │   │
///                       ┌─┴───┴─┐
///                       │  Add  │
///                       └───┬───┘
///                           │
///                       ┌───┴───┐
///                       │ Output│
///                       └───────┘
///
/// After:
///     ┌───────┐  ┌──────────────────────┐  ┌──────────────────────┐  ┌─────────────┐
///     │   Q   │  │Concat(K_cache,K_new) │  │Concat(V_cache,V_new) │  │AttentionMask│
///     └───┬───┘  └──────────┬───────────┘  └──────────┬───────────┘  └──────┬──────┘
///         │                 │                         │                     │
///     ┌───┴─────────────────┴─────────────────────────┴─────────────────────┴─┐
///     │                     ScaledDotProductAttention                         │
///     └─────────────────────────────────┬─────────────────────────────────────┘
///                                       │
///                                  ┌────┴────┐
///                                  │  Output │
///                                  └─────────┘
///
/// K_cache and K_new are concatenated along the sequence axis (and similarly for V).
/// When the matmul transpose flags indicate that K or V are stored with seq as the
/// innermost dimension (i.e. [B,H,D,S]), explicit Transpose nodes are inserted to bring
/// them to the canonical [B,H,S,D] layout required by v13::ScaledDotProductAttention.
/// Scale is set to 1.0 because the matched pattern does not contain a separate scale node;
/// any required scaling is assumed to be pre-applied to Q by the model.
class TRANSFORMATIONS_API SDPASplitAttentionFusionMatcher : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPASplitAttentionFusionMatcher", "0");
    SDPASplitAttentionFusionMatcher();
};

// Temporary wrapper to enable Symbolic infrastructure inside.
class TRANSFORMATIONS_API SDPAFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPAFusion");
    SDPAFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
