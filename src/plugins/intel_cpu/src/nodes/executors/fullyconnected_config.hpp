// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "config.h"
#include "executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

// @todo require explicit initialization of all the attributes?
struct FCAttrs {
    // @todo probably we don't want with bias flag, since this information is already
    // a part of src memory descs
    bool withBias = false;
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    uint64_t dynamicQuantizationGroupSize = 0;
    bool constantWeights = true;

    ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::Config::ModelType::Unknown;

    PostOps postOps;
};

using FCConfig = executor::Config<FCAttrs>;
}  // namespace ov::intel_cpu
