// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"

namespace ov::intel_cpu {

struct MatMulAttrs {
    bool transposeA;
    bool transposeB;
};

using MatMulConfig = executor::Config<MatMulAttrs>;
}  // namespace ov::intel_cpu
