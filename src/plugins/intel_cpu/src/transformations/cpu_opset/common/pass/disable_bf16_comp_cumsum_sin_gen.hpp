// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Keeps the F0 oscillator chain (StyleTTS2 / iSTFTNet's `l_sin_gen`)
 *        in fp32 when the CPU plugin infers in BF16. The chain integrates
 *        phase through CumSum and produces a periodic excitation via Sin;
 *        any BF16 rounding along the phase path drifts the accumulated
 *        phase and audibly distorts the vocoder output.
 *
 *  The pass matches the fixed l_sin_gen core topology below and directly marks
 *  the matched core nodes with @ref ov::disable_conversion(node, f32, bf16).
 *  CPU's EnforceInferencePrecision then expands this into a continuous fp32
 *  island (including the upstream phase-preparation ops) during precision
 *  enforcement.
 *
 *      Transpose -> Interpolate -> Transpose -> CumSum
 *          -> Multiply   -> Transpose -> Multiply
 *          -> Interpolate -> Transpose -> Sin
 *
 *  The pass is intentionally scoped to nodes with `l_sin_gen` in friendly
 *  names to avoid matching unrelated structurally similar chains.
 *
 */
class DisableBF16CompCumSumSinGen : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableBF16CompCumSumSinGen");
    DisableBF16CompCumSumSinGen();
};

}  // namespace ov::intel_cpu
