// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "llm_fc.hpp"

namespace llmdnn {

bool fc_kernel_create_amx(fc_kernel** mm, const fc_create_param* param);

void fc_kernel_destroy_amx(const fc_kernel* mm);

void fc_kernel_execute_amx(const fc_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end, float* dq, float* q, float* bias);

void fc_kernel_bf16w8_get_q_dq_amx(size_t K, size_t N, size_t stride, void* ptr, float* q, float* dq);
void fc_kernel_bf16w8_set_q_dq_amx(const fc_kernel* mm, float q, float dq);

}