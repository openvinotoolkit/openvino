// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "cpu_memory.h"
#include "executor_config.hpp"

namespace ov {
namespace intel_cpu {

struct GEMMAttrs {
    bool withBias = false;
    bool transpose_a = false;
    bool transpose_b = false;
    bool sparseWeights = false;
    std::vector<float> dequantizationScales;
    ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::Config::ModelType::Unknown;
};

}  // namespace intel_cpu
}  // namespace ov
