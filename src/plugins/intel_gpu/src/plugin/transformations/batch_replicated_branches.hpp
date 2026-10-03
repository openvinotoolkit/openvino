// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// Batches N statically indexed replica branches (e.g. camera inputs) before
// their shared convolutional backbone and restores the original per-branch
// concatenated output. N, the per-branch shape, and the concat axis are all
// derived from the matched graph rather than fixed to one reference model.
class BatchReplicatedBranches : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::intel_gpu::BatchReplicatedBranches");
    BatchReplicatedBranches();
};

}  // namespace ov::intel_gpu
