// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "conv_mul_add_fq_block.hpp"
#include "convert_gemm_bias.hpp"
#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ConvertConvolutionBias is the Convolution specialization of ConvertGemmBias
 *     It matches the int8 Convolution requantization chain via ConvMulAddFQBlock
 *
 * Supported patterns:
 *     1. u8 source, u8 or i8 weights
 *     2. i8 source, i8 weights
 *
 * Before:
 *
 * +--------------+    +---------------+
 * | Input (u8/i8)|    | Weights (i8)  |
 * +-----------+--+    +-+-------------+
 *             |         |
 *        +----v---------v----+
 *        |   Convolution     |
 *        +---------+---------+
 *                  |
 *        +-------------------+    +--------------+
 *        |      Multiply     |    | Constant     |
 *        +---------+---------+    +--+-----------+
 *                  |                 |
 *             +----v--------+--------v---+
 *             |           Add            |
 *             +------------+-------------+
 *                          |
 *                   +------v-------+
 *                   | FakeQuantize |
 *                   +------+-------+
 *                          |
 *                  +-------v--------+
 *                  |     Result     |
 *                  +----------------+
 *
 * After:
 *
 * +--------------+    +---------------+
 * | Input (u8/i8)|    | Weights (i8)  |
 * +-----------+--+    +-+-------------+
 *             |         |
 *        +----v---------v----+
 *        |   Convolution     |
 *        +---------+---------+
 *                  |
 *                  |              +--------------+
 *                  |              | Constant     |
 *                  |              +--+-----------+
 *                  |                 |
 *                  |           +-----v------+
 *                  |           |   Round    |
 *                  |           +-----+------+
 *                  |                 |
 *                  |           +-----v------+
 *                  |           | Convert i32|
 *                  |           +-----+------+
 *                  |                 |
 *             +----v--------+--------v----+
 *             |           Add             |
 *             +------------+--------------+
 *                          |
 *                   +------v-------+
 *                   |  Multiply    |
 *                   +------+-------+
 *                          |
 *                   +------v-------+
 *                   | FakeQuantize |
 *                   +------+-------+
 *                          |
 *                  +-------v--------+
 *                  |     Result     |
 *                  +----------------+
 *
 */

namespace ov::intel_cpu {

class ConvertConvolutionBias : public ConvertGemmBias<ConvMulAddFQBlock> {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertConvolutionBias");
    ConvertConvolutionBias() : ConvertGemmBias("ConvertConvolutionBias") {}
};

}  // namespace ov::intel_cpu
