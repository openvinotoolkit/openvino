// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

/*
 * Description:
 *     ConvertConv1DBase detects 1D Convolution / GroupConvolution and replaces
 *     it with the sequence Unsqueeze - 2D Convolution / GroupConvolution - Squeeze.
 *     Unsqueeze adds the additional dimension to Convolution inputs and Squeeze
 *     removes the additional dimension from the Convolution output.
 *
 * Before:
 *
 * +--------------+    +---------------+
 * | Input tensor |    | Kernel tensor |
 * +-----------+--+    +-+-------------+
 *             |         |
 *           +-v---------v-+
 *           | Convolution |
 *           +------+------+
 *                  |
 *           +------v------+
 *           |   Result    |
 *           +-------------+
 *
 * After:
 *
 * +--------------+    +--------------+ +---------------+   +--------------+
 * | Input tensor |    | Constant (1) | | Kernel tensor |   | Constant (1) |
 * +-----------+--+    +-+------------+ +-----------+---+   +-+------------+
 *             |         |                          |         |
 *           +-v---------v-+                      +-v---------v-+
 *           | Unsqueeze   |                      | Unsqueeze   |
 *           +------+------+                      +------+------+
 *                  |                                    |
 *           +------v------------------------------------v------+  +--------------+
 *           |                  Convolution                     |  | Constant (1) |
 *           +---------------------------------------------+----+  +-+------------+
 *                                                         |         |
 *                                                       +-v---------v-+
 *                                                       |   Squeeze   |
 *                                                       +------+------+
 *                                                              |
 *                                                       +------v------+
 *                                                       |    Result   |
 *                                                       +-------------+
 *
 */

namespace ov::intel_cpu {
class ConvertConv1DBase : public ov::pass::MatcherPass {
protected:
    OPENVINO_MATCHER_PASS_RTTI("ConvertConv1DBase");
    template <class Conv>
    ov::matcher_pass_callback convert_conv1d_to_conv2d();
};

class ConvertConv1D : public ConvertConv1DBase {
public:
    OPENVINO_RTTI("ConvertConv1D", "0", ConvertConv1DBase);
    ConvertConv1D();
};

class ConvertGroupConv1D : public ConvertConv1DBase {
public:
    OPENVINO_RTTI("ConvertGroupConv1D", "0", ConvertConv1DBase);
    ConvertGroupConv1D();
};
}  // namespace ov::intel_cpu
