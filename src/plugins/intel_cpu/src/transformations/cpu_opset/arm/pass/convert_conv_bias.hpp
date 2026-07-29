// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ConvertConvolutionBias detects specific quantized Convolution patterns with
 *     Convolution -> Multiply -> Add -> FQ
 *     and inserts a Convert to i32 between the constant bias and the Add node.
 *     Convert to i32 is neccessary because ACL supports i32 bias only.
 *     Also, the order of Add and Multiply is swapped to satisfy ACL requirements.
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

class ConvertConvolutionBias : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertConvolutionBias");
    ConvertConvolutionBias();
};

}  // namespace ov::intel_cpu
