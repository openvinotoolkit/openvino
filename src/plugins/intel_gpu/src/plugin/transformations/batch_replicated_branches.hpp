// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// Batches equivalent replicated branches to improve GPU
// execution by replacing repeated per-replica computation with one batched
// computation.
//
// Before:
//     rank-5 input -> scalar Gather per replica -> equivalent branch -> Concat
//
// After:
//     rank-5 input -> batch Concat -> one batched branch -> Transpose/Reshape
//     restoring the original Concat output layout
//
// The replica count, branch shape, and Concat axis are derived from the graph.
// The pass applies only to graphs with at least two replicas, fully static
// shapes (including all spatial and channel dimensions), supported branch
// operations with matching attributes/runtime metadata, safe broadcast
// semantics, and no unsupported control or data dependencies. Partially
// dynamic shapes, TypeRelaxed branches, and other unsupported graphs are
// intentionally left unchanged.
class BatchReplicatedBranches : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::BatchReplicatedBranches");
    BatchReplicatedBranches();
};

}  // namespace ov::intel_gpu
