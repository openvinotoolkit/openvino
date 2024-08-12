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
    std::vector<float> dequantizationScales;
};

using MatMulConfig = executor::Config<MatMulAttrs>;
}  // namespace intel_cpu
}  // namespace ov
