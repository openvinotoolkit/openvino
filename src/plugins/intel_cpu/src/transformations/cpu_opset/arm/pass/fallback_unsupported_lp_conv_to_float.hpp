// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Converts unsupported quantized Convolution -> Multiply -> Add -> (Swish) ->
 * FakeQuantize patterns to a floating convolution. The fallback applies when
 * activation dequantization contains a zero-point Subtract or the FakeQuantize
 * output precision differs from the quantized Convolution input precision.
 *
 * Dequantization scaling moves from the convolution output to its weights,
 * avoiding overflow of unscaled accumulations in the f16 path. The precision
 * follows the floating Add output: f32 graphs retain f32, while f16 graphs use
 * f16. Inputs, scaled weights and the type-relaxed output use matching types
 * so supported cases select a fast ACL implementation.
 */
namespace ov::intel_cpu {

class FallbackUnsupportedLPConvToFloat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FallbackUnsupportedLPConvToFloat");
    FallbackUnsupportedLPConvToFloat();
};

}  // namespace ov::intel_cpu
