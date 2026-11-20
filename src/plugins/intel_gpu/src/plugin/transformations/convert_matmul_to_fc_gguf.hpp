// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Replaces a MatMul on GGUF Q5_K / Q6_K weights by a FullyConnectedGGUF, which reads the GGUF blocks directly.
 *
 * Without this pass, constant folding decodes the GGUFDequantize into a full-size floating-point constant.
 *
 * Before:                                        After:
 *
 *     Constant u8 [N, B]
 *            │
 *    ┌───────┴──────┐
 *    │GGUFDequantize│ Q5_K / Q6_K
 *    └───────┬──────┘
 *            │
 *       [Convert]                                A     Constant u8 [N, B]
 *            │                                   │        │
 *     A      │                                ┌──┴────────┴───────┐
 *     │      │                                │FullyConnectedGGUF │
 *   ┌─┴──────┴─┐                              └───────────────────┘
 *   │  MatMul  │ transpose_b
 *   └──────────┘
 *
 * B = K / 256 * block_bytes. Q6_K blocks are regrouped into planes (see op::FullyConnectedGGUF).
 * Applied when the GGUFDequantize has no other consumer and the MatMul takes the weights transposed (one GGUF row per
 * output).
 */
class ConvertMatMulToFullyConnectedGGUF : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertMatMulToFullyConnectedGGUF");
    ConvertMatMulToFullyConnectedGGUF();
};

}  // namespace ov::intel_gpu
