// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps the F0 oscillator chain (StyleTTS2 / iSTFTNet's `l_sin_gen`)
 *        in fp32 by tagging it with `disable_fp16_compression`. The chain
 *        cumulatively integrates phase via CumSum and applies sin() at the
 *        predicted F0 (voice pitch); fp16 loses precision on the accumulated
 *        phase and produces audible noise in the F0 / first-formant band.
 *
 *  Before                                  After
 *  ------                                  -----
 *  any (FP16)  ──┐                         any (FP32)  ──┐    ◄─ also tagged
 *                ▼                                       ▼
 *             CumSum     (FP16)                       CumSum     (FP32)
 *                │                                       │
 *                ▼                                       ▼
 *             Multiply   (FP16)                       Multiply   (FP32)
 *                │                                       │
 *                ▼                                       ▼
 *             Transpose  (FP16)                       Transpose  (FP32)
 *                │                                       │
 *                ▼                                       ▼
 *             Multiply   (FP16)                       Multiply   (FP32)
 *                │                                       │
 *                ▼                                       ▼
 *             Interpolate(FP16)                       Interpolate(FP32)
 *                │                                       │
 *                ▼                                       ▼
 *             Transpose  (FP16)                       Transpose  (FP32)
 *                │                                       │
 *                ▼                                       ▼
 *               Sin      (FP16) ── audible noise        Sin      (FP32) ── clean
 *                │                                       │
 *                ▼                                       ▼
 *           downstream                             downstream
 *
 *  Note: only `disable_fp16_compression` rt_info is added; no graph topology
 *  change. The downstream `ConvertPrecision` pass uses the rt_info to keep
 *  the marked nodes in fp32 while the rest of the model is lowered to fp16.
 */
class DisableFP16CompCumSumSinGen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompCumSumSinGen");
    DisableFP16CompCumSumSinGen();
};
}  // namespace ov::intel_gpu
