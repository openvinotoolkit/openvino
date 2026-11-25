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
//  - gamma scales the normalized axis only: static shape {..., 1, N}, no more dims than x.
//
// The pass runs before ConvertPrecision, so the fusion removes the Divide that MarkDivWithEps would mark to keep
// the norm and, through propagation, its neighbors in FP32. The pass sets these marks itself:
//  - the RMS is marked to stay in FP32 when that mark, propagated up from x, reaches a residual connection
//    (elementwise arithmetic of two non-constant inputs, e.g. the residual Add) or a real-typed model input, which
//    may carry a residual stream: ConvertPrecision then keeps the residual stream in FP32, which may exceed the f16
//    range. Other norms (e.g. q/k norms fed by a MatMul) run in f16;
//  - the RMS is also marked to stay in FP32 when a node that the fusion removes already has that mark (e.g. from the
//    model's rt_info), since copy_runtime_info does not copy it;
//  - the scaled gamma is marked to stay in FP32: it may be dequantized from u16 values above the f16 range
//    (e.g. 65535). KeepDequantizationPrecision covers Constant weights only, not a u16 weight fed as a model input
//    (as ONNX Runtime with the OpenVINO EP passes it). The Converts that dequantize such a non-constant gamma are
//    marked as well: the FP32 mark propagates up through a Convert only to a Constant.
class L2NormToRMS : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("L2NormToRMS");
    explicit L2NormToRMS(size_t max_work_group_size);
};

}  // namespace ov::intel_gpu
