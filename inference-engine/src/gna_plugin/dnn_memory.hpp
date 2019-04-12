// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// dnn_memory.hpp : memory manipulation routines

#pragma once

#include <cstdint>
extern void MemoryAssign(void **ptr_dest,
                         void **ptr_memory,
                         uint32_t num_bytes_needed,
                         uint32_t *ptr_num_bytes_used,
                         uint32_t num_memory_bytes,
                         const char *name);
