// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "executor_config.hpp"

namespace ov::intel_cpu {

struct GroupedMatMulAttrs {
    uint64_t dynamic_quantization_group_size = 0;
};

using GroupedMatMulConfig = executor::Config<GroupedMatMulAttrs>;

}  // namespace ov::intel_cpu
