// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     Converts an unsupported quantized Conv -> Multiply -> Add -> (Swish) ->
 *     FakeQuantize pattern to a floating convolution. The pass applies when:
 *       1. The Conv activation is produced by Subtract (zero-point dequantization), or
 *       2. The FakeQuantize output type differs from the Conv activation type.
 *
 * Before (the two supported activation branches):
 *
 *     Case 1: X(u8/i8) -> [Convert] -> Subtract(zero_point) -> A
 *     Case 2: X(u8/i8) -> [Convert] -----------------------> A
 *
 *     [Convert] is optional. The Subtract, when present, is preserved.
 *     For u8 X, weights W may be u8/i8; for i8 X, W must be i8.
 *
 *     +--------------+       +--------------+
 *     | Activation A |       |  Weights W   |
 *     +------+-------+       +------+-------+
 *            |                      |
 *            +----------+-----------+
 *                       v
 *                +-------------+
 *                | Convolution |
 *                +------+------+
 *                       |
 *                +------v------+       +-------------+
 *                |  Multiply   | <---- | DQ scales S |
 *                +------+------+       +-------------+
 *                       |
 *                +------v------+       +-------------+
 *                |     Add     | <---- | Float bias  |
 *                +------+------+       +-------------+
 *                       |
 *                   (Swish)
 *                       |
 *                +------v------+
 *                | FakeQuantize|
 *                +-------------+
 *
 * After:
 *
 *     +--------------+       +--------------+       +-------------+
 *     | Activation A |       |  Weights W   |       | DQ scales S |
 *     +------+-------+       +------+-------+       +------+------+
 *            |                      |                      |
 *       Convert(T)             Convert(T)             Convert(T)
 *            |                      |                      |
 *            |                      |              Reshape[C,1,...,1]
 *            |                      |                      |
 *            |                +-----v------+               |
 *            |                |  Multiply  | <-------------+
 *            |                +-----+------+
 *            |                      |
 *            +----------+-----------+
 *                       v
 *                +-------------+
 *                | Convolution |  (activation, scaled weights and output: T)
 *                +------+------+
 *                       |
 *                +------v------+       +-------------+
 *                |     Add     | <---- | Float bias  |
 *                +------+------+       +-------------+
 *                       |
 *                   (Swish)
 *                       |
 *                +------v------+
 *                | FakeQuantize|
 *                +-------------+
 *
 *     For f16/f32 floating graphs, T is the Add output type: f32 stays f32,
 *     f16 stays f16. Convert(T) is inserted only when the input type differs.
 *     The type-relaxed Conv output override is updated to T as well. This avoids
 *     mixed f16 source/weights with f32 destination, which selects reference Conv.
 *
 *     Per-output-channel scaling moves from the Conv result to its weights:
 *         Conv(A, W) * S  ->  Conv(A, W * Reshape(S)).
 *     C is 1 for per-tensor scales or OC for per-output-channel scales.
 *     This prevents large unscaled Conv outputs from overflowing in the f16 path.
 *     Add, optional Swish and FakeQuantize remain in the same order; the bias is
 *     not multiplied by S. Floating-point rounding can differ after this rewrite.
 */

namespace ov::intel_cpu {

class FallbackUnsupportedLPConvToFloat : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FallbackUnsupportedLPConvToFloat");
    FallbackUnsupportedLPConvToFloat();
};

}  // namespace ov::intel_cpu
