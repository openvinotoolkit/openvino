// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "unordered_map"

namespace ov::intel_cpu {

using MemoryDescArgs = std::unordered_map<int, MemoryDescPtr>;
using MemoryArgs = std::unordered_map<int, MemoryPtr>;

// basic inputs
#define ARG_SRC_0 1
#define ARG_SRC   ARG_SRC_0
#define ARG_SRC_1 2
#define ARG_SRC_2 3
#define ARG_DST_0 17
#define ARG_DST   ARG_DST_0
#define ARG_WEI_0 33
#define ARG_WEI   ARG_WEI_0
#define ARG_BIAS  41
// legacy dequantization scale
#define ARG_DST_DEQ_SCALE 53
// scaling factors provided at execution time
#define ARG_ATTR_SCALES 4096
// zero points provided at execution time
#define ARG_ATTR_ZERO_POINTS 8192

}  // namespace ov::intel_cpu
