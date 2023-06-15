// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cstring>
#include <thread>

#include "llm_mm.hpp"

namespace llmdnn {

bool mm_kernel_create_amx(mm_kernel** mm, const mm_create_param* param);

void mm_kernel_destroy_amx(const mm_kernel* mm);

void mm_kernel_execute_amx(const mm_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K);

}