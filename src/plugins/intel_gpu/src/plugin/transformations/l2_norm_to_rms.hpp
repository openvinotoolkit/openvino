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
class L2NormToRMS : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("L2NormToRMS");
    explicit L2NormToRMS(size_t max_work_group_size);
};

// Rewrites elementwise v1::Power(x, 0.5) (with an optional Convert on the constant) to v0::Sqrt(x)
// when output shape and element type are preserved, enabling downstream RMSFusion / MVNFusion /
// NormalizeL2Fusion / L2NormToRMS patterns that match v0::Sqrt.
class ConvertPowerHalfToSqrt : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ConvertPowerHalfToSqrt");
    ConvertPowerHalfToSqrt();
};

}  // namespace ov::intel_gpu
