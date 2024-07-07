// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"

namespace ov {
namespace intel_cpu {

struct MatMulAttrs {
    bool transposeA;
    bool transposeB;
    // @todo only memory descriptors should be a part of attributes
    // actual memory should be passed into "execute" or "prepareMemory" calls
    std::vector<float> dequantizationScales;
};

using MatMulConfig = executor::Config<MatMulAttrs>;
}  // namespace intel_cpu
}  // namespace ov
