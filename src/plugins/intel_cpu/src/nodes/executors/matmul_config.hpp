// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"

namespace ov {
namespace intel_cpu {

// @todo require explicit initialization of all the attributes?
struct MatMulAttrs {
    bool transposeA;
    bool transposeB;
    // @todo only memory descriptors should be a part of attributes
    // actual memory should be passed into "execute" or "prepareMemory" calls
    std::vector<float> dequantizationScales;
    int64_t activation_k_dim;
    int64_t activation_offset;
};

using MatMulConfig = executor::Config<MatMulAttrs>;
}  // namespace intel_cpu
}  // namespace ov
