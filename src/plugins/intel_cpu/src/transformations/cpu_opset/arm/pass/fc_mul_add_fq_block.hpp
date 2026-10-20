// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/op/block.hpp"

/*
 * Description:
 *     FCMulAddFQBlock is a reusable pattern block that matches the int8 FullyConnected requantization chain:
 *         MatMul -> Multiply -> Add -> FakeQuantize
 *
 *     The MatMul activation and weights inputs are i8 (the int8 FullyConnected case lowered by LPT). The bias Add
 *     consumes a non-i32 constant; the FakeQuantize requantizes the result back to i8/u8.
 */

namespace ov::intel_cpu {

class FCMulAddFQBlock : public ov::pass::pattern::op::Block {
public:
    explicit FCMulAddFQBlock(bool require_int_fq_output);
};

}  // namespace ov::intel_cpu
