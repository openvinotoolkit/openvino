// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convert_gemm_bias.hpp"
#include "fc_mul_add_fq_block.hpp"
#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ConvertFullyConnectedBias is the FullyConnected specialization of ConvertGemmBias
 *     It matches the int8 FullyConnected requantization chain via FCMulAddFQBlock
 *
 * Before:  MatMul -> Multiply -> Add(bias f16/f32) -> FakeQuantize -> Result
 * After:   MatMul -> Add(Round(bias)->Convert i32) -> Multiply -> FakeQuantize -> Result
 *
 * Supported patterns:
 *     1. u8 source, i8 weights
 *     2. i8 source, i8 weights
 */

namespace ov::intel_cpu {

class ConvertFullyConnectedBias : public ConvertGemmBias<FCMulAddFQBlock> {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFullyConnectedBias");
    ConvertFullyConnectedBias() : ConvertGemmBias("ConvertFullyConnectedBias") {}
};

}  // namespace ov::intel_cpu
