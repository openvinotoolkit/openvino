// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/op/block.hpp"

/*
 * Description:
 *     ConvMulAddFQBlock is a reusable pattern block that matches:
 *         (optional) Convert -> (optional) Subtract -> Convolution -> Multiply -> Add -> FakeQuantize
 *
 *     The Convolution activation input may be:
 *       - i8 (with i8 weights)
 *       - u8 (with i8 or u8 weights)
 *       - optional Convert and Subtract (zero-point dequantization)
 */

namespace ov::intel_cpu {

class ConvMulAddFQBlock : public ov::pass::pattern::op::Block {
public:
    explicit ConvMulAddFQBlock(bool require_int_fq_output);
};

}  // namespace ov::intel_cpu
