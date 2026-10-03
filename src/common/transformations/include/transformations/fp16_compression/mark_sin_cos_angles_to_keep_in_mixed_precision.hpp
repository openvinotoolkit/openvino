// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass {
class TRANSFORMATIONS_API MarkSinCosAnglesToKeepInMixedPrecision;
}  // namespace ov::pass

/**
 * @ingroup ov_transformation_common_api
 * @brief Keeps the computation of Sin/Cos arguments that are derived from model inputs (positions,
 * coordinates, timesteps) and constants only in the original precision.
 *
 * Positional encodings computed at runtime, such as decomposed RoPE tables, multiply position
 * indices by frequencies before Sin/Cos, so the arguments reach magnitudes of ~1e4 rad. Low
 * precision types quantize such values in steps larger than 2*pi (f16: 8 rad, bf16: 64 rad at
 * 1.6e4), which turns the resulting tables into noise. Sin/Cos themselves are not affected,
 * only the arithmetic that produces their argument.
 *
 * Every node on the input path of a matched Sin/Cos, up to Parameters, Constants and ShapeOfs, is
 * marked with disable_conversion(f16), which is honored both by ConvertPrecision and by the CPU
 * plugin's bf16 enforcement:
 *
 *     positions         freqs              positions         freqs
 *         │               │                    │               │
 *   ┌─────┴──────┐        │              ┌─────┴──────┐        │
 *   │ arithmetic │        │              │ arithmetic │ (f32)  │
 *   │  & layout  ├────────┘              │  & layout  ├────────┘
 *   └─────┬──────┘                       └─────┬──────┘
 *      ┌──┴──┐                              ┌──┴──┐
 *      │ Sin │                              │ Sin │ (f32)
 *      └─────┘                              └─────┘
 *
 * The path is left unmarked when it reaches a MatMul or a convolution: the argument then depends on
 * activations, so keeping it in f32 would cost performance without addressing a positional table.
 * Angle tables computed by an outer-product MatMul (LLM-style RoPE) are covered after RoPE fusion by
 * MarkRopeInputsToKeepInMixedPrecision.
 */
class ov::pass::MarkSinCosAnglesToKeepInMixedPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("MarkSinCosAnglesToKeepInMixedPrecision");
    MarkSinCosAnglesToKeepInMixedPrecision();
};
