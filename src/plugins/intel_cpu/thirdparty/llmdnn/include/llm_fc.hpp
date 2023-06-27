// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "llm_types.hpp"

namespace llmdnn {

typedef enum {
    NONE = 0,
    DEQUANT = 1 << 0,
    BIAS = 1 << 1,
    GELU = 1 << 2,
    QUANT = 1 << 3,

    BIAS_GELU = BIAS | GELU,
    DEQUANT_BIAS_GELU = DEQUANT | BIAS_GELU,
    DEQUANT_BIAS_GELU_QUANT = DEQUANT_BIAS_GELU | QUANT,
    DEQUANT_BIAS_QUANT = DEQUANT | BIAS | QUANT,
    DEQUANT_GELU_QUANT = DEQUANT | GELU | QUANT,
    DEQUANT_QUANT = DEQUANT | QUANT,

    DEQUANT_GELU = DEQUANT | GELU,
    DEQUANT_BIAS = DEQUANT | BIAS
} postops_types;

struct fc_create_param {
    data_type_t dt_a;
    data_type_t dt_b;
    data_type_t dt_c;
    bool b_is_trans;
    postops_types postops_type;
    // for weight compression
    float q;
    float dq;
};

struct fc_kernel;

/// Generates a mm kernel based on param
///
/// @param mm Output kernel
/// @param param kernel parameters, supported:
///        fc: (s8,s8,s8),dq,[bias],[gelu],q
///        fc: (s8,s8,bf16),dq,[bias],[gelu]
///        fc: (s8,s8,f32),dq,[bias],[gelu]
///        fc: (bf16,bf16,bf16),[bias],[gelu]
///        fc: (bf16,bf16,f32),[bias],[gelu]
///        fc: (bf16,s8,f32),dq,[bias],[gelu]
///        fc: (bf16,s8,bf16),dq,[bias],[gelu]
///
bool fc_kernel_create(fc_kernel** mm, const fc_create_param* param);
void fc_kernel_destroy(const fc_kernel* mm);
void fc_kernel_execute(const fc_kernel* mm,
        void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end,
        float* dq=nullptr, float* q=nullptr, float* bias=nullptr);

/// weight compression
/// compute weight min/max once, set q, dq for each fc_kernel instance
void fc_kernel_bf16w8_get_q_dq(size_t K, size_t N, size_t stride, void* ptr, float* q, float* dq);

}
