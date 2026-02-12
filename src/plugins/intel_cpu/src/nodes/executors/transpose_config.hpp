// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "executor_config.hpp"
#include "memory_arguments.hpp"
#include "nodes/common/permute_kernel.h"

namespace ov::intel_cpu {

struct TransposeParams {
    PermuteParams permuteParams;
};

struct TransposeAttrs {
    TransposeParams params;
    MemoryDescArgs descs;
};

using TransposeConfig = executor::Config<TransposeAttrs>;

}  // namespace ov::intel_cpu
