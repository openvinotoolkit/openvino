// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ExcludeMaxPoolPadding detects MaxPool operations with explicit non-zero
 *     padding, creates a single Pad operation with the full extended
 *     pads_begin/pads_end vectors, and rewires MaxPool to consume that Pad.
 *     The MaxPool node is then updated to use zero paddings with
 *     PadType::VALID, so the padding is represented explicitly in the graph.
 *     The Pad value is chosen as 0 when the MaxPool input comes from a
 *     FakeQuantize with non-negative output range, otherwise it is set to the
 *     lowest representable floating-point value of the input type.
 *     This form allows the ARM ACL backend to enable exclude_padding-based
 *     optimized MaxPool kernels while keeping padding on all spatial axes in a
 *     single explicit Pad node.
 *
 * Before:
 *
 * +---------------+
 * | Input tensor  |
 * +-------+-------+
 *         |
 *   +-----v--------------------------------+
 *   | MaxPool                              |
 *   | pads_begin != 0 or pads_end != 0     |
 *   | auto_pad = EXPLICIT                  |
 *   +------------------+-------------------+
 *                      |
 *               +------v------+
 *               |   Result    |
 *               +-------------+
 *
 * After:
 *
 * +---------------+    +------------------+    +------------------+    +------------------+
 * | Input tensor  |    | pads_begin full  |    | pads_end full    |    | pad value        |
 * +-------+-------+    +--------+---------+    +--------+---------+    | 0 or type min    |
 *         |                     |                       |              +---------+--------+
 *         |                     |                       |                        |
 *   +-----v---------------------v-----------------------v------------------------v---+
 *   | Pad (CONSTANT)                                                                 |
 *   +-----------------------------------+--------------------------------------------+
 *                                       |
 *                             +---------v----------------+
 *                             | MaxPool                  |
 *                             | pads_begin = 0           |
 *                             | pads_end = 0             |
 *                             | auto_pad = VALID         |
 *                             +------------+-------------+
 *                                          |
 *                                   +------v------+
 *                                   |   Result    |
 *                                   +-------------+
 *
 */

namespace ov::intel_cpu {

class ExcludeMaxPoolPadding : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ExcludeMaxPoolPadding");
    ExcludeMaxPoolPadding();
};

}  // namespace ov::intel_cpu