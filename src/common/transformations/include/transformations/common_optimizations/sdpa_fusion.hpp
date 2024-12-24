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
class TRANSFORMATIONS_API SDPAFusion : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAFusion", "0");
    SDPAFusion();
};

}  // namespace pass
}  // namespace ov
