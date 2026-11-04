// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

// An RMS norm written as an L2 normalization over the last axis, the form ONNX LpNormalization(p=2)
// decomposes to:
//     x / Sqrt(ReduceSum(x^2) + eps) * gamma
// Since ReduceSum = N * ReduceMean over N elements, this is exactly
//     RMS(x, gamma / sqrt(N), eps / N)
// RMSFusion does not take it: it looks for ReduceMean and requires gamma to be a Constant, while here
// gamma may be computed at run time (e.g. a dequantized weight fed as a model input).
// skip_residual_input (for runs before ConvertPrecision): skip norms fed by a residual Add, so MarkDivWithEps
// still keeps the residual stream in FP32. The scaled gamma is always kept in FP32 (u16 gamma may exceed f16).
class L2NormToRMS : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("L2NormToRMS");
    explicit L2NormToRMS(size_t max_work_group_size, bool skip_residual_input = false);
};

}  // namespace ov::intel_gpu
