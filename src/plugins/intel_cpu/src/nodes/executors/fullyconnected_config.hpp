// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "config.h"
#include "executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

// @todo require explicit initialization of all the attributes?
struct FCAttrs {
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    uint64_t dynamicQuantizationGroupSize = 0;
    bool constantWeights = true;

    ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::Config::ModelType::Unknown;

    // Per-channel (or per-tensor) dequantization scales folded from the post-FC dequantization Multiply by
    // GraphOptimizer::FuseConvMatmulFCDeconvAndDQScales (ARM int8). Mirrors ConvAttrs::dqScales; consumed by the
    // ACL int8 FullyConnected executor as the weights requantization scale.
    std::vector<float> dqScales;

    PostOps postOps;
};

using FCConfig = executor::Config<FCAttrs>;
}  // namespace ov::intel_cpu
