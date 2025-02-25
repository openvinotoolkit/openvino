// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cpu_memory.h"
#include "executor_config.hpp"

namespace ov::intel_cpu {

// @todo require explicit initialization of all the attributes?
struct FCAttrs {
    // @todo probably we don't want with bias flag, since this information is already
    // a part of src memory descs
    bool withBias = false;
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    uint64_t dynamicQuantizationGroupSize;

    ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::Config::ModelType::Unknown;
};

using FCConfig = executor::Config<FCAttrs>;
}  // namespace ov::intel_cpu
