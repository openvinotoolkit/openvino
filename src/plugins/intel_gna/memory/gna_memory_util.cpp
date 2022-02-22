// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gna_memory_util.hpp"

#include <cstdint>
#include "gna_plugin_log.hpp"

int32_t GNAPluginNS::memory::MemoryOffset(void *ptr_target, void *ptr_base) {
    auto target = reinterpret_cast<uintptr_t>(ptr_target);
    auto base = reinterpret_cast<uintptr_t>(ptr_base);
    if (target == 0) {  // handle NULL pointers separately
        return (-1);
    } else if (target < base) {
        THROW_GNA_EXCEPTION << "Target address value " <<  target << " is less than base address " << base;
    } else {
        uint64_t diff = target - base;
        if (diff > 0x7fffffff) {
            THROW_GNA_EXCEPTION << "Target address value " << target << " too far from base address " << base;
        }
        return static_cast<int32_t>(diff);
    }
}

