// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cinttypes>
#include <cstring>
#include <cstdint>

#ifdef _WIN32
#include <malloc.h>
#else
#include <mm_malloc.h>
#endif

#include <details/ie_exception.hpp>

#include "gna_plugin_log.hpp"
#include "gna_memory_util.hpp"


void * GNAPluginNS::memory::AllocateMemory(uint32_t num_memory_bytes, const char *ptr_name) {
    void *ptr_memory = _mm_malloc(num_memory_bytes, 64);
    if (ptr_memory == NULL) {
        THROW_GNA_EXCEPTION << "Memory allocation failed for " << ptr_name;
    }
    memset(ptr_memory, 0, num_memory_bytes);

    return (ptr_memory);
}

void GNAPluginNS::memory::FreeMemory(void *ptr_memory) {
    if (ptr_memory != NULL) {
        _mm_free(ptr_memory);
    }
    ptr_memory = NULL;
}

int32_t GNAPluginNS::memory::MemoryOffset(void *ptr_target, void *ptr_base) {
    uint64_t target = (uint64_t) ptr_target;
    uint64_t base = (uint64_t) ptr_base;
    if (target == 0) {  // handle NULL pointers separately
        return (-1);
    } else if (target < base) {
        THROW_GNA_EXCEPTION << "Error:  target address value " <<  target<< " is less than base address " << base << " in MemoryOffset()";
    } else {
        uint64_t diff = target - base;
        if (diff > 0x7fffffff) {
            THROW_GNA_EXCEPTION << "Error:  target address value " << target << " too far from base address " << base << " in MemoryOffset()!";
        }
        return ((int32_t) diff);
    }
}

