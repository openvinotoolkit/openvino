// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdio>
#include <cstdlib>
#include "dnn_memory.hpp"
#include "gna-api.h"

void MemoryAssign(void **ptr_dest,
                  void **ptr_memory,
                  uint32_t num_bytes_needed,
                  uint32_t *ptr_num_bytes_used,
                  uint32_t num_memory_bytes,
                  const char *name) {
    if (*ptr_num_bytes_used + ALIGN(num_bytes_needed, 64) > num_memory_bytes) {
        fprintf(stderr,
                "Out of memory in %s (%d+ALIGN(%d)>%d)!\n",
                name,
                *ptr_num_bytes_used,
                num_bytes_needed,
                num_memory_bytes);
        throw -1;
    } else {
        uint8_t *ptr_bytes = reinterpret_cast<uint8_t *>(*ptr_memory);
        *ptr_dest = *ptr_memory;
        *ptr_memory = ptr_bytes + ALIGN(num_bytes_needed, 64);
        *ptr_num_bytes_used += ALIGN(num_bytes_needed, 64);
    }
}
