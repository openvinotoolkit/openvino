// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ConvertConvolutionBias detects specific quantized Convolution patterns followed by Multiply and Add
 *     and inserts a Convert to i32 between the constant bias and the Add node.
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
 *           +------v------+
 *           |   Multiply  |
 *           +------+------+
 *                  |
 *                  |              +--------------+
 *                  |              | Constant     |
 *                  |              +--+-----------+
 *                  |                 |
 *             +----v--------+--------v----+
 *             |           Add            |
 *             +------------+-------------+
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
 *           +------v------+
 *           |   Multiply  |
 *           +------+------+
 *                  |
 *                  |              +--------------+
 *                  |              | Constant     |
 *                  |              +--+-----------+
 *                  |                 |
 *                  |           +-----v------+
 *                  |           | Convert i32|
 *                  |           +-----+------+
 *                  |                 |
 *             +----v--------+--------v----+
 *             |           Add            |
 *             +------------+-------------+
 *                          |
 *                  +-------v--------+
 *                  |     Result     |
 *                  +----------------+
 *
 */

namespace ov::intel_cpu {

class ConvertConvolutionBias : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertConvolutionBias");
    ConvertConvolutionBias();
};

}  // namespace ov::intel_cpu
