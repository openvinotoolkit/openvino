// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_types.hpp"

namespace llmdnn {

struct mm_create_param {
    data_type_t dt_a;
    data_type_t dt_b;
    bool b_is_gemv;     // true if matrix b is vector. Shape: a[M,K], b[K,1], c[M,1]
    bool b_is_trans;
};

struct mm_kernel;

/// Generates a mm kernel based on param
///
/// @param mm Output kernel
/// @param param kernel parameters, supported:
///        matmul: (u8/s8,s8,f32)
///        gemv: (s8,s8,f32)
///        matmul: (bf16,bf16,f32)
///        gemv: (bf16,bf16,f32)
///
bool mm_kernel_create(mm_kernel** mm, const mm_create_param* param);
void mm_kernel_destroy(const mm_kernel* mm);

void mm_kernel_execute(const mm_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K);

}
