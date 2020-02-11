// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

namespace GNAPluginNS {
namespace memory {

void *AllocateMemory(uint32_t num_memory_bytes, const char *ptr_name);
void FreeMemory(void *ptr_memory);
int32_t MemoryOffset(void *ptr_target, void *ptr_base);

}  // namespace memory
}  // namespace GNAPluginNS
