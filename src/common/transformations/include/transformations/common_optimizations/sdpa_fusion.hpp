// Copyright (C) 2024 Intel Corporation
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

class TRANSFORMATIONS_API SDPAFusionMatcherSinks : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAFusionMatcherSinks", "0");
    SDPAFusionMatcherSinks();
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
