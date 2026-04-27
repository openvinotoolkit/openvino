// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     AlignUnsupportedLPConvFQPrecision detects quantized Convolution patterns with
 *     Convolution -> Multiply -> Add -> FakeQuantize and retargets the FakeQuantize
 *     output precision to the Convolution activation precision.
 *
 *     ARM ACL low-precision convolution supports only the same low-precision type
 *     on the activation input and the post-convolution FakeQuantize output. When the
 *     FakeQuantize output type differs between u8 and i8, later ARM-specific passes
 *     skip the pattern and the convolution falls back to fp16.
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
 *             | act=u8/i8   |
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
 *             +------------------------------+
 *             | FakeQuantize                 |
 *             | out=i8/u8 (mismatch)         |
 *             +------+-----------------------+
 *                    |
 *                    v
 *             +-------------+
 *             |   Result    |
 *             +-------------+
 *
 * After:
 *
 * +--------------+      +---------------+
 * | Input (u8/i8)|      | Weights (i8)  |
 * +------+-------+      +-------+-------+
 *        |                      |
 *        +----------+-----------+
 *                   |
 *             +-----v------+
 *             | Convolution |
 *             | act=u8/i8   |
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
 *             +------------------------------+
 *             | FakeQuantize                 |
 *             | out=Convolution act type     |
 *             | PrecisionsAttribute={act}    |
 *             +------+-----------------------+
 *                    |
 *                    v
 *             +-------------+
 *             |   Result    |
 *             +-------------+
 */

namespace ov::intel_cpu {

class AlignUnsupportedLPConvFQPrecision : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AlignUnsupportedLPConvFQPrecision");
    AlignUnsupportedLPConvFQPrecision();
};

}  // namespace ov::intel_cpu