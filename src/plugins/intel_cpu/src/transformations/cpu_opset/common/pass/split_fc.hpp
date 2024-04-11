// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov {
namespace intel_cpu {

/*
 * Description:
 *      SplitFC detects FC CPU operation with and without compressed weights.
 *      And then splits the FC into several small FCs by output channel according to sub stream number.
 *      The goal is that the executor can dispatch the split FCs to different numa nodes in the system.
 *      As a result, the split FCs can be executed at the parallel level.
 *
 * Before:
 *
 *             +-------+                         +-------+
 *             |   X   |                         |   W   |
 *             |       |                         |       |
 *             |       |                         |       |
 *             +-------+                         +-------+
 *                 |                                 |
 *                 |                                 |
 * +---------------v---------------------------------v--------------+
 * |                                                                |
 * |                        FullyConnected                          |
 * |                                                                |
 * +------------------------------+---------------------------------+
 *                                |
 *                                | Output
 *                                v
 *
 * After:
 *
 *            +-------+                           +-------+
 *            |   X   |                           |   W   |
 *            |       |                           |       |
 *            |       |                           |       |
 *            +---+---+                           +---+---+
 *                |                                   |
 *                |                                   |
 *                |                           +-------v-------+
 *                |                           |               |
 *                |                           | VariadicSplit |
 *                |                           |               |
 *                |                           +--+---------+--+
 *                |                              |         |
 *                |     +------------------------+         |
 *                |     |                                  |
 *            +---------|------------------------+         |
 *            |         |                        |         |
 * +----------v---------v---------+  +-----------v---------v--------+
 * |                              |  |                              |
 * |        FullyConnected        |  |        FullyConnected        |
 * |                              |  |                              |
 * +--------------+---------------+  +--------------+---------------+
 *                |                                 |
 *                | Output                          | Output
 *                |                                 |
 * +--------------v---------------------------------v---------------+
 * |                                                                |
 * |                            Concat                              |
 * |                                                                |
 * +-------------------------------+--------------------------------+
 *                                 |
 *                                 |
 *                                 v
 */

class SplitFC: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("SplitFC", "0");
    SplitFC(int sub_stream_num);
};

}   // namespace intel_cpu
}   // namespace ov
