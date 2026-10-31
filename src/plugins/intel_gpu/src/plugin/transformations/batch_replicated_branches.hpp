// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// Runs N equal replicated branches (e.g. per-camera backbones) as one batched branch.
//
// Before:
//     data -> Gather(i) -> Convolution -> [MaxPool] -> [Add] -> [Relu] -> Concat(axis > 0), i = 0..N-1
// After:
//     data -> Gather(0..N-1) -> Concat(axis 0) -> Convolution -> [MaxPool] -> [Add] -> [Relu]
//          -> Transpose -> Reshape
//
// Branches must be static [1, C, H, W], feed only the Concat, and have equal attributes and constants.
class BatchReplicatedBranches : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::BatchReplicatedBranches");
    BatchReplicatedBranches();
};

}  // namespace ov::intel_gpu
