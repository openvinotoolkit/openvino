// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     FallbackUnsupportedLPConvToFP16 detects quantized Convolution patterns with
 *     Convolution -> Multiply -> Add -> FakeQuantize that cannot execute as int8.
 *
 *     The pass fires when either:
 *       1. The Convolution activation comes from a Subtract (zero-point dequantization),
 *          indicating the int8 path is broken for this Conv (unconditional fallback).
 *       2. The FakeQuantize output precision differs from Conv activation precision
 *          (original type-mismatch case).
 *
 *     The pass moves DQ scaling from Convolution output path to Convolution weights:
 *     post-conv DQ Multiply is removed, and equivalent scaling is applied on weights
 *     before Convolution.
 *     This avoids fp16 overflow on large post-conv values that would otherwise be scaled
 *     in the output path.
 *
 * Before (case 1 - Subtract on activation):
 *
 *  +--------+     +----------+
 *  |Convert | --> | Subtract | (zero-point)    +---------------+
 *  +--------+     +-----+----+                 | Weights (i8)  |
 *                       |                      +-------+-------+
 *                       +----------+-----------+
 *                                  |
 *                            +-----v------+
 *                            | Convolution |
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            |  Multiply   |
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            |     Add     |
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            | FakeQuantize|
 *                            +------+------+
 *                                   |
 *                                   v
 *                            +-------------+
 *                            |   Result    |
 *                            +-------------+
 *
 * Before (case 2 - type mismatch, no Subtract):
 *
 * +--------------+      +---------------+
 * | Input (u8/i8)|      | Weights (i8)  |
 * +------+-------+      +-------+-------+
 *        |                      |
 *        +----------+-----------+
 *                   |
 *             +-----v------+
 *             | Convolution |
 *             +------+------+
 *                    |
 *                    v
 *             +-------------+
 *             |  Multiply   |
 *             +------+------+
 *                    |
 *                    v
 *             +-------------+
 *             |     Add     |
 *             +------+------+
 *                    |
 *                    v
 *             +-------------+
 *             | FakeQuantize|  (output precision != Conv activation precision)
 *             +------+------+
 *                    |
 *                    v
 *             +-------------+
 *             |   Result    |
 *             +-------------+
 *
 * After:
 *
 * +--------------+      +----------------+                             +--------------+
 * | Input (u8/i8)| ---> | Convert (f16)  | --------------------------> |              |
 * +--------------+      +----------------+                             | Convolution  |
 *                                                                      |              |
 * +---------------+      +----------------+      +------------------+  | (f16 act,    |
 * | Weights (i8)  | ---> | Convert (f16)  | ---> |     Multiply     |->| scaled f16   |
 * +---------------+      +----------------+      +--------+---------+  | weights)     |
 *                                                         ^            +------+-------+
 *                                                         |                   |
 * +--------------+      +----------------+      +---------+--------+          v
 * | DQ Scales    | ---> | Convert (f16)  | ---> |    Reshape       |   +------+------+
 * +--------------+      +----------------+      +------------------+   |     Add     |
 *                                                                      +------+------+
 *                                                                             |
 *                                                                             v
 *                                                                      +------+------+
 *                                                                      | FakeQuantize|
 *                                                                      +------+------+
 *                                                                             |
 *                                                                             v
 *                                                                      +------+------+
 *                                                                      |   Result    |
 *                                                                      +-------------+
 *
 * Note:
 *     Convert(f16) nodes are inserted only when the corresponding input is not f16.
 */

namespace ov::intel_cpu {

class FallbackUnsupportedLPConvToFP16 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FallbackUnsupportedLPConvToFP16");
    FallbackUnsupportedLPConvToFP16();
};

}  // namespace ov::intel_cpu
