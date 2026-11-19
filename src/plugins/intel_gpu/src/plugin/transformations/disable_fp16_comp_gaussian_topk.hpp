// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/**
 * @brief Keeps the threshold of a Gaussian top-k activation sparsification in FP32.
 *        The threshold is mean(x) + k * std(x), with std decomposed as
 *        sqrt(mean((x - mean(x))^2)). In FP16 the squared deviation overflows to Inf
 *        once |x - mean(x)| > 256, which turns the threshold into Inf and x - threshold
 *        into -Inf; the subsequent ReLU then zeroes every element in that row.
 *
 *        Matched sub-graph (Gemma-3n MLP layers 0-9); x, mean and deviation each feed several nodes:
 *
 *                                                     x  (e.g. gate_proj MatMul output)
 *                                                     |
 *                +------------------------------------+---------------------------------+
 *                |                                    |                                 |
 *                v                                    v                                 |
 *   +-------------------------+    +-------------------------------------+              |
 *   | ReduceMean "mean"       |    | Subtract "deviation"                |              |
 *   | in0 = x                 |    | in0 = x                             |              |
 *   | axes = {-1}, keep_dims  |--->| in1 = mean                          |              |
 *   +-------------------------+    +-------------------------------------+              |
 *                |                            |               |                         |
 *                |                            v               v                         |
 *                |                 +-------------------------------------+              |
 *                |                 | Multiply "squared_deviation"        |              |
 *                |                 | in0 = deviation, in1 = deviation    |              |
 *                |                 +-------------------------------------+              |
 *                |                                    v                                 |
 *                |                 +-------------------------------------+              |
 *                |                 | ReduceMean "variance"               |              |
 *                |                 | in0 = squared_deviation             |              |
 *                |                 | axes = {-1}, keep_dims              |              |
 *                |                 +-------------------------------------+              |
 *                |                                    v                                 |
 *                |                 +-------------------------------------+              |
 *                |                 | Sqrt "std_dev"                      |              |
 *                |                 | in0 = variance                      |              |
 *                |                 +-------------------------------------+              |
 *                |                                    v                                 |
 *                |                 +-------------------------------------+              |
 *                |                 | Multiply "scaled_std_dev"           |    +-----+   |
 *                |                 | in0 = std_dev, in1 = k              |<---|  k  |   |
 *                |                 +-------------------------------------+    +-----+   |
 *                |                                    v                                 |
 *                |                 +-------------------------------------+              |
 *                |                 | Add "threshold"                     |              |
 *                +---------------->| in0 = mean, in1 = scaled_std_dev    |              |
 *                                  +-------------------------------------+              |
 *                                                     v                                 |
 *                                  +-------------------------------------+              |
 *                                  | Subtract "sparsified"               |              |
 *                                  | in0 = x, in1 = threshold            |<-------------+
 *                                  +-------------------------------------+
 *                                                     |
 *                                                     v
 *                                                 ReLU (not part of the pattern)
 *
 *        k = erfinv(0.9) * sqrt(2) = 1.6448536 is a scalar computed from constants (ErfInv, Multiply) and
 *        shared by all layers; the pattern accepts any node as k.
 *
 * FP16 conversion is disabled (ov::disable_conversion) for the squared deviation and the nodes
 * computed from it (Multiply, ReduceMean, Sqrt, Multiply, Add). ConvertPrecision with
 * keep_precision_sensitive_in_fp32 propagates the mark up to mean and deviation. The graph
 * topology is unchanged.
 */
class DisableFP16CompForGaussianTopK : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForGaussianTopK");
    DisableFP16CompForGaussianTopK();
};

}  // namespace ov::intel_gpu
