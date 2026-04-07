// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/matcher_pass.hpp"

/*
 * Description:
 *     ExcludeMaxPoolPadding detects MaxPool operations with explicit non-zero
 *     padding and moves that padding into a dedicated Pad operation.
 *     The MaxPool node is then updated to use zero paddings with
 *     PadType::VALID, so the padding is represented explicitly in the graph.
 *     This form allows the ARM ACL backend to enable exclude_padding-based
 *     optimized MaxPool kernels.
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
 * +---------------+    +------------------+    +------------------+
 * | Input tensor  |    | pads_begin/end   |    | -inf converted   |
 * +-------+-------+    +--------+---------+    | to input type     |
 *         |                     |              +---------+--------+
 *         |                     |                        |
 *   +-----v---------------------v------------------------v---+
 *   | Pad (CONSTANT)                                        |
 *   +--------------------------+----------------------------+
 *                              |
 *                    +---------v----------------+
 *                    | MaxPool                  |
 *                    | pads_begin = 0           |
 *                    | pads_end = 0             |
 *                    | auto_pad = VALID         |
 *                    +------------+-------------+
 *                                 |
 *                          +------v------+
 *                          |   Result    |
 *                          +-------------+
 *
 */

namespace ov::intel_cpu {

class ExcludeMaxPoolPadding : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ExcludeMaxPoolPadding");
    ExcludeMaxPoolPadding();
};

}  // namespace ov::intel_cpu