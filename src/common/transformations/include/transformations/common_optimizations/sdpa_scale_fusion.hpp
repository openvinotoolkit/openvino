// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

/// Merges explicit multiplication by scalar value for Q and K into scale attribute of SDPA op
/// Before:
///     ┌───────┐    ┌───────┐    ┌───────┐  ┌─────────────┐     ┌─────────────┐
///     │   Q   │    │   K   │    │   V   │  │AttentionMask│     │    Scale    |
///     └───┬───┘    └───┬───┘    └───┬───┘  │ (Optional)  │     │  (Optional) │
///         │            │            │      └──────┬──────┘     └───────┬─────┘
///         │            │            │             │                    |
///     ┌───┴───┐    ┌───┴───┐        │             │                    |
///     │  Mul  |    │  Mul  │        |             │                    |
///     └───┬───┘    └───┬───┘        │             │                    │
///         │            │            │             │                    │
///         |            │            │             │                    │
///     ┌───┴────────────┴────────────┴─────────────┴─┐                  |
///     │           ScaledDotProductAttention         │──────────────────┘
///     └────────────────────┬────────────────────────┘
///                          │
///                          │
///                     ┌────┴────┐
///                     │  Output │
///                     └─────────┘
/// After:
///     ┌───────┐    ┌───────┐    ┌───────┐  ┌─────────────┐  ┌───────┐
///     │   Q   │    │   K   │    │   V   │  │AttentionMask│  │ Scale |
///     └───┬───┘    └───┬───┘    └───┬───┘  └──────┬──────┘  └───┬───┘
///         │            │            │             │             |
///         │            │            │             │             |
///         |            │            │             │             |
///     ┌───┴────────────┴────────────┴─────────────┴─┐           |
///     │           ScaledDotProductAttention         │───────────┘
///     └────────────────────┬────────────────────────┘
///                          │
///                          │
///                     ┌────┴────┐
///                     │  Output │
///                     └─────────┘
/// Multiply ops for Q and K are eliminated in the following cases:
/// 1. Q_scale and K_scale are constant
/// 2. Q_scale * SDPA_Scale == 1 or K_scale * SDPA_Scale == 1
class TRANSFORMATIONS_API SDPAScaleFusionPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAScaleFusionPass", "0");
    SDPAScaleFusionPass();
};

class TRANSFORMATIONS_API SDPAScaleFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPAScaleFusion");
    SDPAScaleFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
