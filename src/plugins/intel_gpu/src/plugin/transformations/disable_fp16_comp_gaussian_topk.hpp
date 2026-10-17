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
 *          x ----------+------------------------------+
 *          |           |                              |
 *     ReduceMean   Subtract(x, mean)                  |
 *          |           |                              |
 *          |      Multiply(d, d)                      |
 *          |           |                              |
 *          |      ReduceMean -> Sqrt -> Multiply(k)   |
 *          |                              |           |
 *          +----------------> Add <-------+           |
 *                              |                      |
 *                          Subtract(x, threshold) <---+
 *
 * Only disable_fp16_compression rt_info is added to the threshold computation
 * (ReduceMean, Subtract, Multiply, ReduceMean, Sqrt, Multiply, Add); the graph
 * topology is unchanged.
 */
class DisableFP16CompForGaussianTopK : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("DisableFP16CompForGaussianTopK");
    DisableFP16CompForGaussianTopK();
};

}  // namespace ov::intel_gpu
