// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ExtractConvActivationZeroPoint matches an activation zero-point Subtract that
 *     directly feeds a Convolution/GroupConvolution and removes it from the graph,
 *     storing the zero-point value in the Convolution's rt_info instead
 *     (key: ExtractConvActivationZeroPoint::rt_info_key). This lets the ACL executor
 *     apply the zero-point natively via QuantizationInfo instead of paying for a
 *     separate Subtract op.
 *
 *     Pattern matched: Activation -> Subtract(zero_point) -> Convolution/GroupConvolution
 *
 *     Two behaviors depending on activation precision and what follows the Convolution:
 *
 *     1. Direct elision (i8 activation, or u8 activation with a matching u8 tail and no
 *        Swish): the zero point is already in ACL's native domain for that dtype, so it
 *        is moved into rt_info unchanged.
 *
 *     2. u8 -> i8 shift (u8 activation, everything else - i.e. i8 tail or a
 *        dequantize-to-F32 tail): ACL's i8/dequantize kernels only accept a signed i8
 *        source, so the u8 domain is remapped onto i8 by shifting the producing
 *        FakeQuantize's output_low/output_high bounds by -128 and overriding its output
 *        type to i8. The stored zero point is shifted to match (zp - 128), preserving
 *        the original math: (x - 128) - (zp - 128) = x - zp.
 *
 * Before (case 1 - direct elision, e.g. i8 activation):
 *
 *  +--------------+     +----------------+
 *  | Activation   |     | Zero-point     |
 *  | (i8)         |     | (Constant, i8) |
 *  +-------+------+     +-------+--------+
 *          |                    |
 *          +---------+----------+
 *                     |
 *               +-----v------+     +---------------+
 *               |  Subtract  |     | Weights       |
 *               +-----+------+     +-------+-------+
 *                     |                    |
 *                     +---------+----------+
 *                               |
 *                     +---------v----------+
 *                     | Convolution /      |
 *                     | GroupConvolution   |
 *                     +--------------------+
 *
 * After (case 1):
 *
 *  +--------------+                        +---------------+
 *  | Activation   |                        | Weights       |
 *  | (i8)         |                        +-------+-------+
 *  +-------+------+                                |
 *          |                                       |
 *          +------------------+--------------------+
 *                             |
 *                   +---------v----------+
 *                   | Convolution /      |
 *                   | GroupConvolution   |
 *                   | rt_info[key] = zp  |
 *                   +--------------------+
 *
 * Before (case 2 - u8 activation, i8/dequant tail):
 *
 *  +--------------------------+     +----------------+
 *  | FakeQuantize             |     | Zero-point     |
 *  | out: u8                  |     | (Constant, u8) |
 *  | bounds: [lo, hi]         |     +-------+--------+
 *  +-------------+------------+             |
 *                +-------------+------------+
 *                              |
 *                        +-----v------+     +---------------+
 *                        |  Subtract  |     | Weights       |
 *                        +-----+------+     +-------+-------+
 *                              |                    |
 *                              +---------+----------+
 *                                        |
 *                              +---------v----------+
 *                              | Convolution /      |
 *                              | GroupConvolution   |
 *                              +---------+----------+
 *                                        |
 *                                        v
 *                          (i8 FakeQuantize, or dequant Multiply -> F32)
 *
 * After (case 2):
 *
 *  +--------------------------+                       +---------------+
 *  | FakeQuantize             |                       | Weights       |
 *  | out: i8 (overridden)     |                       +-------+-------+
 *  | bounds: [lo-128, hi-128] |                               |
 *  +-------------+------------+                               |
 *                +------------------+-------------------------+
 *                                   |
 *                         +---------v-------------+
 *                         | Convolution /          |
 *                         | GroupConvolution       |
 *                         | rt_info[key] = zp - 128|
 *                         +------------------------+
 */

namespace ov::intel_cpu {

class ExtractConvActivationZeroPoint : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ExtractConvActivationZeroPoint");
    ExtractConvActivationZeroPoint();

    static constexpr const char* rt_info_key = "activation_zero_point";
};

}  // namespace ov::intel_cpu
