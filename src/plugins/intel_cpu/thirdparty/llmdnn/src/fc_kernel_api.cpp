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
#include <map>

#include "llm_fc.hpp"
#include "fc_kernel_amx.hpp"
#include "mm_kernel_common_amx.hpp"
#include "utility_avx512.hpp"

namespace llmdnn {

static decltype(&fc_kernel_create) fc_kernel_create_ptr = fc_kernel_create_amx;
static decltype(&fc_kernel_destroy) fc_kernel_destroy_ptr = fc_kernel_destroy_amx;
static decltype(&fc_kernel_execute) fc_kernel_execute_ptr = fc_kernel_execute_amx;
static decltype(&fc_kernel_bf16w8_get_q_dq) fc_kernel_bf16w8_get_q_dq_ptr = fc_kernel_bf16w8_get_q_dq_amx;
static decltype(&fc_kernel_bf16w8_set_q_dq) fc_kernel_bf16w8_set_q_dq_ptr = fc_kernel_bf16w8_set_q_dq_amx;

// interface
bool fc_kernel_create(fc_kernel** mm, const fc_create_param* param) {
    return fc_kernel_create_ptr(mm, param);
}

void fc_kernel_destroy(const fc_kernel* mm) {
    fc_kernel_destroy_ptr(mm);
}

void fc_kernel_execute(const fc_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end, float* dq, float* q, float* bias) {
    fc_kernel_execute_ptr(mm, ptr_a, ptr_b, ptr_c, lda, ldb, ldc, M, N, K, n_start, n_end, dq, q, bias);
}

void fc_kernel_bf16w8_get_q_dq(size_t K, size_t N, size_t stride, void* ptr, float* q, float* dq) {
    fc_kernel_bf16w8_get_q_dq_ptr(K, N, stride, ptr, q, dq);
}

/// set q, dq for each fc_kernel instance
void fc_kernel_bf16w8_set_q_dq(const fc_kernel* mm, float q, float dq) {
    fc_kernel_bf16w8_set_q_dq_ptr(mm, q, dq);
}

}