// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_mm.hpp"
#include "mm_kernel_amx.hpp"

namespace llmdnn {

static decltype(&mm_kernel_create) mm_kernel_create_ptr = mm_kernel_create_amx;
static decltype(&mm_kernel_destroy) mm_kernel_destroy_ptr = mm_kernel_destroy_amx;
static decltype(&mm_kernel_execute) mm_kernel_execute_ptr = mm_kernel_execute_amx;

// interface
bool mm_kernel_create(mm_kernel** mm, const mm_create_param* param) {
    return mm_kernel_create_ptr(mm, param);
}

void mm_kernel_destroy(const mm_kernel* mm) {
    mm_kernel_destroy_ptr(mm);
}

void mm_kernel_execute(const mm_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K) {
    mm_kernel_execute_ptr(mm, ptr_a, ptr_b, ptr_c, lda, ldb, ldc, M, N, K);
}

}