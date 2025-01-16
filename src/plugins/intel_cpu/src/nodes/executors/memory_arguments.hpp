// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc.h"
#include "unordered_map"

namespace ov {
namespace intel_cpu {

using MemoryDescArgs = std::unordered_map<int, MemoryDescPtr>;
using MemoryArgs     = std::unordered_map<int, MemoryPtr>;

// @todo add more options
#define ARG_SRC_0 1
#define ARG_SRC   ARG_SRC_0
#define ARG_SRC_1 2
#define ARG_SRC_2 3
#define ARG_DST_0 17
#define ARG_DST   ARG_DST_0
#define ARG_WEI_0 33
#define ARG_WEI   ARG_WEI_0
#define ARG_BIAS  41

}  // namespace intel_cpu
}  // namespace ov
