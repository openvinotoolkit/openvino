// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"
#include "nodes/common/permute_kernel.h"

namespace ov::intel_cpu {

struct TransposeAttrs {
    PermuteParams permuteParams;
};

using TransposeConfig = executor::Config<TransposeAttrs>;

}  // namespace ov::intel_cpu
