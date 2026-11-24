// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

/**
 * @brief Normalizes eligible rank-3 LINEAR_ONNX Interpolate operations
 *        to rank-4 for GPU execution.
 *
 * Before: [N,C,L] -> Interpolate -> [N,C,Lout]
 * After:  [N,C,L] -> Unsqueeze(2) -> [N,C,1,L]
 *                -> Interpolate(axes={2,3}) -> [N,C,1,Lout]
 *                -> Squeeze(2) -> [N,C,Lout]
 *
 * The input rank must be known; its dimensions may be dynamic.
 * Axes must be constant or omitted and include the last axis. Omitted axes default
 * to all input axes; explicit empty axes are left unchanged. Constant positive
 * sizes or finite positive scales are required to prove that N and C are unchanged
 * before excluding them from interpolation. Sizes on N/C require static dimensions
 * for that proof. Padding and antialiasing are excluded because promoting rank
 * alone does not establish equivalent spatial-kernel behavior for those cases.
 * Both v4 and v11 are handled without changing the op version.
 * Replacement scales/sizes are constructed directly as a same-type Constant,
 * so legalization does not depend on constant-folding evaluator availability.
 */
class NormalizeInterpolateRank3 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("NormalizeInterpolateRank3");
    NormalizeInterpolateRank3();
};

}  // namespace ov::intel_gpu
