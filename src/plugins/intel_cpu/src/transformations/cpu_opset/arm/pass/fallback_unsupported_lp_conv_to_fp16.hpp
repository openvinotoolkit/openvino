// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     FallbackUnsupportedLPConvToFP16 detects quantized Convolution patterns with
 *     Convolution -> Multiply -> Add -> FakeQuantize, when dequantized
 *     FakeQuantize output precision is different from Convolution activation precision.
 *     This precision-mismatch case is not supported by ACL int8 executor and is
 *     handled by convolution fp primitive.
 *     The pass moves DQ scaling from Convolution output path to Convolution weights:
 *     post-conv DQ Multiply is removed, and equivalent scaling is applied on weights
 *     before Convolution.
 *     This avoids fp16 overflow on large post-conv values that would otherwise be scaled
 *     in the output path.
 *
 * Before:
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
 *             | FakeQuantize|
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
