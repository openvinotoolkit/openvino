// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Converts Unsqueeze + Unsqueeze + Multiply + ReduceSum patterns into MatMul.
///
/// Matches the pattern where two N Dims inputs are each unsqueezed on different axes,
/// broadcast-multiplied to produce an (N+1) Dims intermediate, and then reduced by
/// ReduceSum on a third axis that is neither of the unsqueeze axes. This is
/// mathematically equivalent to a batched MatMul with optional transposes.
///
/// Example (Mamba2 SSD naive):
///   A:[b,g,L,H,N] --Unsqueeze(3)--> [b,g,L,1,H,N]
///   B:[b,g,L,H,N] --Unsqueeze(2)--> [b,g,1,L,H,N]
///   Multiply --> [b,g,L,L,H,N]
///   ReduceSum(axis=5) --> [b,g,L,L,H]
///   == Batched MatMul: [b*g*H, L, N] x [b*g*H, N, L] -> [b*g*H, L, L]
class BroadcastMulReduceToMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("BroadcastMulReduceToMatMul");
    BroadcastMulReduceToMatMul();
};

}   // namespace ov::intel_gpu
