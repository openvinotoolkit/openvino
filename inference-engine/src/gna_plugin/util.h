// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

void *AllocateMemory(uint32_t num_memory_bytes, const char *ptr_name);
void FreeMemory(void *ptr_memory);
int32_t MemoryOffset(void *ptr_target, void *ptr_base);
