// Copyright (C) 2020-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ConvertConvolutionDQScales detects quantized Convolution patterns with
 *     Convolution -> Multiply -> Add -> FakeQuantize, when dequantized
 *     FakeQuantize output precision is different from Convolution activation precision.
 *     Precision check is required to ensure this case can't be handled by ACL executor.
 *
 * Supported patterns:
 *     1. u8 source, u8 or i8 weights
 *     2. i8 source, i8 weights
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

class ConvertConvolutionDQScales : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertConvolutionDQScales");
    ConvertConvolutionDQScales();
};

}  // namespace ov::intel_cpu
