// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/type/element_type.hpp"
#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Applies activations scaling around ov::intel_gpu::op::MOECompressed.
 *
 * For a scale factor s, this pass:
 *   - inserts Multiply(1/s) (and Convert to scaled_prec if needed) in front of
 *     the hidden_states input (index 0) and the two bias inputs (bias_up at
 *     index 4 and bias_down at index 6 for the has_zp=false layout);
 *   - records s on the MOECompressed op via set_input_scale(s). The op builder
 *     propagates this to the internal SwiGLU primitive, which multiplies its
 *     inputs by s to restore the original range before clamp / swish / up_add.
 *
 * This is the MOECompressed analogue of ov::pass::activations_scaling::ScaleDownSingleLayer,
 * which handles MatMul / Convolution.
 */
class ScaleDownMOECompressed : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::ScaleDownMOECompressed");

    ScaleDownMOECompressed(float scale_factor, ov::element::Type scaled_prec);
};

}  // namespace ov::intel_gpu
