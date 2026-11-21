// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_cpu {

/**
 * @brief Makes the divisor of AvgPool(CEIL_TORCH, exclude_pad=false) count only positions inside the explicitly
 * padded input, as ONNX and PyTorch define it.
 *
 * With ceil rounding the last window can extend past `in + pads_end`. oneDNN's include-padding mode divides by the
 * full kernel size, so that extra area is counted. The pass rewrites the pool as
 *     AvgPool(x, exclude_pad=true) * (real positions / padded positions)
 * where exclude_pad semantics agree on every executor. The ratio only depends on shapes and is constant-folded
 * for static spatial dimensions. Pools whose windows never extend past the padding are left unchanged.
 */
class AvgPoolCeilTorchDivisor : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("AvgPoolCeilTorchDivisor");
    AvgPoolCeilTorchDivisor();
};

}  // namespace ov::intel_cpu
