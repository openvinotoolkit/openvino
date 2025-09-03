// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

struct MatMulAttrs {
    bool transposeA = false;
    bool transposeB = false;
    bool withBias = false;
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    uint64_t dynamicQuantizationGroupSize = 0;
    bool constantWeights = false;
    bool fcSemantic = false;

    // DQ scales for quantization
    std::vector<float> dqScales;

    // Post-operations for fused operations
    PostOps postOps;
};

using MatMulConfig = executor::Config<MatMulAttrs>;
}  // namespace ov::intel_cpu
