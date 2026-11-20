// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

// Fuses an RMS norm written as an L2 normalization over the last axis, the form ONNX LpNormalization(p=2)
// decomposes to, into internal::RMS. Since ReduceSum = N * ReduceMean over N elements:
//     x / Sqrt(ReduceSum(x^2) + eps) * gamma == RMS(x, gamma / sqrt(N), eps / N)
// RMSFusion does not take it: it looks for ReduceMean and requires gamma to be a Constant, while here
// gamma may be computed at run time (e.g. a dequantized weight fed as a model input).
//
// Before:                                  After:
//
//          x                                  x    gamma   1/sqrt(N)
//          |                                  |       \     /
//   [Convert] (optional)                      |      Multiply
//      |        |                             |         |
//      |      Power(2)                        +---RMS---+  (eps / N)
//      |        |                                  |
//      |    ReduceSum(last axis, keep_dims)   output type of the matched root
//      |        |
//      |      Add(eps)
//      |        |
//      |      Sqrt
//      |        |
//      +--Divide
//           |
//        Multiply--gamma
//           |
//     [Convert] (optional: a sole consumer of the Multiply, RMS writes its type directly)
//
// Conditions:
//  - static N within the RMS kernel limit (N / 8 <= max_work_group_size, as RMSFusion on GPU);
//    rank <= 5 (the kernel has bfyx and bfzyx implementations only);
//  - ReduceSum, Add(eps), Sqrt and Divide have no other consumers; eps is a scalar Constant;
//  - gamma scales the normalized axis only: static shape {..., 1, N}, no more dims than x;
//  - with skip_residual_input (registration before ConvertPrecision), x is not a residual Add (an Add of two
//    non-constant inputs). Such norms are left to the late registration: MarkDivWithEps keys on the Divide and
//    keeps the residual stream in FP32, and an early fusion would remove the Divide and let it drop to f16.
// The scaled gamma is marked to stay in FP32: it may be dequantized from u16 values above the f16 range
// (e.g. 65535), and after an early fusion no Divide is left to keep it in FP32. KeepDequantizationPrecision covers
// Constant weights only, not a u16 weight fed as a model input (as ONNX Runtime with the OpenVINO EP passes it).
class L2NormToRMS : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("L2NormToRMS");
    explicit L2NormToRMS(size_t max_work_group_size, bool skip_residual_input = false);
};

}  // namespace ov::intel_gpu
