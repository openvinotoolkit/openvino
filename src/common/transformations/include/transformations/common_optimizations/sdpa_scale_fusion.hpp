// Copyright (C) 2018-2026 Intel Corporation
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
/// A constant scalar Multiply is also found when up to two single-consumer Reshape/Squeeze/
/// Unsqueeze/Transpose ops separate it from the SDPA input, since scaling commutes with them
/// (e.g. Q = Transpose(Squeeze(Split(QKV) * scale))); those ops are rebuilt without the Multiply.
class TRANSFORMATIONS_API SDPAScaleFusionPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAScaleFusionPass", "0");
    SDPAScaleFusionPass();
};

/// Merges a constant scalar Multiply on V into a constant scalar Multiply consuming the SDPA output,
/// since the output is linear in V. The output Multiply is removed when the merged scale is 1, as for
/// activations scaling (V * s, SDPA_out * 1/s).
/// Before:
///     V -> Mul(a) -> [up to two Reshape/Squeeze/Unsqueeze/Transpose] -> SDPA -> Mul(b)
/// After:
///     V -> [same data-movement ops] -> SDPA -> Mul(a * b)      (no Mul if a * b == 1)
/// Not applied when a Multiply could change the rank of its data or when a * b is not a finite,
/// normal value of the output element type (it would overflow or flush to zero).
class TRANSFORMATIONS_API SDPAValueScaleFusionPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("SDPAValueScaleFusionPass", "0");
    SDPAValueScaleFusionPass();
};

class TRANSFORMATIONS_API SDPAScaleFusion : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("SDPAScaleFusion");
    SDPAScaleFusion() = default;
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;
};

}  // namespace pass
}  // namespace ov
