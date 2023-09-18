// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pass/graph_rewrite.hpp>

/*
 * Description:
 *     ConvertGroupConvolution detects GroupConvolution and replaces it
 *     with a set of Convolution operations. Number of Convolution operations
 *     equals to number of groups.
 *
 * Before:
 * 
 * +--------------+    +---------------+
 * | Input tensor |    | Kernel tensor |
 * +-----------+--+    +-+-------------+
 *             |         |
 *        +----v---------v----+
 *        | Group Convolution |
 *        +---------+---------+
 *                  |
 *           +------v------+
 *           |   Result    |
 *           +-------------+
 * 
 * After:
 * 
 * +--------------+    +--------------+ +---------------+   +--------------+
 * | Input tensor |    | Constant (1) | | Kernel tensor |   | Constant (0) |
 * +-----------+--+    +-+------------+ +-----------+---+   +-+------------+
 *             |         |                          |         |
 *           +-v---------v-+                      +-v---------v-+
 *           |   Split     |                      |   Split     |
 *           +-+-----------+--------+             +-+---------+-+
 *             |                    |               |         |
 *             |                    |   +-----------v--+    +-v------------+
 *             |                    |   |  Squeeze     |    |   Squeeze    |
 *             |         +----------+---+--------------+    +-+------------+
 *             |         |          |                         |
 *             |         |          +---------------+         |
 *             |         |                          |         |
 * +-----------v---------v------------+ +-----------v---------v------------+
 * |          Convolution             | |          Convolution             |
 * +-----------------------------+----+ +---+------------------------------+
 *                               |          |
 *                         +-----v----------v------+
 *                         |       Concat          |
 *                         +----------+------------+
 *                                    |
 *                         +----------v------------+
 *                         |        Result         |
 *                         +-----------------------+
 * 
 */

namespace ov {
namespace intel_cpu {

class ConvertGroupConvolution: public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ConvertGroupConvolution", "0");
    ConvertGroupConvolution();
};
}  // namespace intel_cpu
}  // namespace ov
