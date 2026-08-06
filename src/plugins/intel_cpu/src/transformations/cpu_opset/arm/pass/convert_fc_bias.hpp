// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ConvertFullyConnectedBias detects the int8 FullyConnected requantization pattern
 *     MatMul -> Multiply -> Add(bias) -> FakeQuantize
 *     and inserts a Convert to i32 between the constant bias and the Add node.
 *     Convert to i32 is necessary because the ACL int8 FullyConnected executor supports i32 bias only.
 *     Also, the order of Add and Multiply is swapped to satisfy ACL requirements (the dequantization Multiply
 *     must directly follow the GEMM, with the bias added as an i32 input). This mirrors ConvertConvolutionBias.
 *
 * Before:  MatMul -> Multiply -> Add(bias f16/f32) -> FakeQuantize -> Result
 * After:   MatMul -> Add(Round(bias)->Convert i32) -> Multiply -> FakeQuantize -> Result
 *
 * Supported patterns:
 *     1. u8 source, i8 weights
 *     2. i8 source, i8 weights
 */

namespace ov::intel_cpu {

class ConvertFullyConnectedBias : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertFullyConnectedBias");
    ConvertFullyConnectedBias();
};

}  // namespace ov::intel_cpu
