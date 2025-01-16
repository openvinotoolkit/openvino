// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cpu_memory.h"
#include "executor_config.hpp"

namespace ov {
namespace intel_cpu {

// @todo require explicit initialization of all the attributes?
struct FCAttrs {
    // @todo probably we don't want with bias flag, since this information is already
    // a part of src memory descs
    bool withBias = false;
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    // @todo only memory descriptors should be a part of attributes
    // actual memory should be passed into "execute" or "prepareMemory" calls
    std::vector<float> dequantizationScales;
    // @todo should be passed as an additional memory input?
    MemoryCPtr decompressionSubtractPtr;
    MemoryCPtr decompressionMultiplyPtr;
    uint64_t dynamicQuantizationGroupSize;
    ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::Config::ModelType::Unknown;
};

using FCConfig = executor::Config<FCAttrs>;
}  // namespace intel_cpu
}  // namespace ov
