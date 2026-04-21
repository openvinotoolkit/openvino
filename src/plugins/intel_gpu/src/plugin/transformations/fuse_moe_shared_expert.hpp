// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"

namespace ov::intel_gpu {

/// Fuse shared expert subgraph (gate/up/down MatMuls + optional sigmoid gating)
/// into the MOE node as additional inputs, absorbing the surrounding Add.
///
/// Before:
///   Add(MOE(hidden, routing, topk, gate, up, down),
///       Reshape?(Mul?(Sigmoid(MatMul(x, gate_gate)), MatMul(Mul(Swish(MatMul(x, sh_gate)), MatMul(x, sh_up)), sh_down))))
///
/// After:
///   MOE(hidden, routing, topk, gate, up, down, sh_gate, sh_up, sh_down, gate_gate)
///
/// This pass should run BEFORE ConvertMOEToMOECompressed so that the compressed
/// conversion pass handles weight extraction uniformly for all MOE weight inputs.
class FuseMOESharedExpert : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("FuseMOESharedExpert");
    FuseMOESharedExpert();
};

}  // namespace ov::intel_gpu
