// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Promotes the Multiply(real, Power(imag, -1)) -> Atan subgraph
 *        (the post-ConvertDivide form of the atan2 decomposition) to f32 and
 *        inserts a Select(imag == 0, 0, mul_out) guard before Atan to avoid
 *        the f16 0/0 -> NaN case. f16 is restored after Atan.
 *
 *  Before                        After
 *  ------                        -----
 *  imag (FP16)   real (FP16)                    imag (FP16) ─────────────────┐
 *     │             │                              │                         │
 *     ▼             │                          Convert(f32)                  │
 *  Power(-1) FP16   │                              │                         ▼
 *     │             │                              ▼                  Equal(imag, 0)
 *     └─► Multiply (FP16)        real (FP16)   Power(-1) FP32                │
 *               │                     │            │                         │
 *               ▼                 Convert(f32)     │                         │
 *           Atan (FP16) ── NaN        │            ▼                         │
 *               │                     └────►   Multiply (FP32)               │
 *               ▼                                  │                         │
 *           downstream                             ▼                         │
 *                                             Select(is_zero, 0, mul) ◄──────┘
 *                                                  │
 *                                                  ▼
 *                                              Atan (FP32)
 *                                                  │
 *                                             Convert(f16)
 *                                                  │
 *                                                  ▼
 *                                              downstream
 */
class IncreaseAtan2DivPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("IncreaseAtan2DivPrecision");
    IncreaseAtan2DivPrecision();
};

}  // namespace ov::intel_gpu
