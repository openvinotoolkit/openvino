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
#include "mm_kernel_common_amx.hpp"
#include "utility_kernel_avx512.hpp"
#include "fc_kernel_amx.hpp"

namespace llmdnn {

using ov::bfloat16;
struct fc_kernel {
    std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>> bf16xbf16;
    std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, int8_t>> bf16xi8;
    std::shared_ptr<amx_kernel::Matmul<int8_t, int8_t>> i8xi8;
    std::shared_ptr<amx_kernel::Matmul<uint8_t, int8_t>> u8xi8;

    data_type_t dt_a;
    data_type_t dt_b;
    data_type_t dt_c;
    postops_types postops_type;
    bool b_is_transpose;
};

using supported_key = std::tuple<data_type_t, data_type_t, data_type_t>;
using supported_value = std::pair<size_t, size_t>;
static std::map<supported_key, supported_value> supported_postops = {
    { { dnnl_s8, dnnl_s8, dnnl_s8 }, { DEQUANT | QUANT, BIAS | GELU } },
    { { dnnl_s8, dnnl_s8, dnnl_bf16 }, { DEQUANT, BIAS | GELU } },
    { { dnnl_s8, dnnl_s8, dnnl_f32 }, { DEQUANT, BIAS | GELU } },
    { { dnnl_bf16, dnnl_bf16, dnnl_bf16 }, { 0, BIAS | GELU } },
    { { dnnl_bf16, dnnl_bf16, dnnl_f32 }, { 0, BIAS | GELU } },
    { { dnnl_bf16, dnnl_s8, dnnl_f32 }, { DEQUANT, BIAS | GELU } },
    { { dnnl_bf16, dnnl_s8, dnnl_bf16 }, { DEQUANT, BIAS | GELU } },
};

static bool check_valid_postops(size_t value, data_type_t dt_a, data_type_t dt_b, data_type_t dt_c) {
    auto it = supported_postops.find(std::make_tuple(dt_a, dt_b, dt_c));
    if (it == supported_postops.end()) {
        return false;
    }
    
    size_t must_have;
    size_t opt_have;
    must_have = (*it).second.first;
    opt_have = (*it).second.second;

    if ((value & must_have) != must_have)
        return false;
    // value must in must_have and opt_have
    if ((value & ~(must_have | opt_have)) != 0)
        return false;
    
    return true;
}

// interface
bool fc_kernel_create_amx(fc_kernel** mm, const fc_create_param* param) {
    fc_kernel* m = nullptr;
    if (param == nullptr || mm == nullptr) {
        std::cout << "fc_kernel_create: invalid input parameter.\n";
        goto ERR;
    }

    if (!check_valid_postops(static_cast<size_t>(param->postops_type), param->dt_a, param->dt_b, param->dt_c)) {
        std::cout << "fc_kernel_create: unsupported data type, a: " << param->dt_a <<", b: " << param->dt_b << ", c: " << param->dt_c <<
            ", postops type: " << param->postops_type << ".\n";
        goto ERR;
    }

    m = new fc_kernel;
    if (param->dt_a == dnnl_s8 && param->dt_b == dnnl_s8) {
        m->i8xi8 = std::make_shared<amx_kernel::Matmul<int8_t, int8_t>>(true, param->b_is_trans);
    } else if (param->dt_a == dnnl_u8 && param->dt_b == dnnl_s8) {
        m->u8xi8 = std::make_shared<amx_kernel::Matmul<uint8_t, int8_t>>(true, param->b_is_trans);
    } else if (param->dt_a == dnnl_bf16 && param->dt_b == dnnl_bf16) {
        m->bf16xbf16 = std::make_shared<amx_kernel::Matmul<bfloat16, bfloat16>>(true, param->b_is_trans);
    } else if (param->dt_a == dnnl_bf16 && param->dt_b == dnnl_s8) {
        m->bf16xi8 = std::make_shared<amx_kernel::Matmul<bfloat16, int8_t>>(true, param->b_is_trans);
    } else {
        std::cout << "fc_kernel_create: unsupport input type, a: " << param->dt_a << ", b: " << param->dt_b << ".\n";
        goto ERR;
    }

    m->dt_a = param->dt_a;
    m->dt_b = param->dt_b;
    m->dt_c = param->dt_c;
    m->b_is_transpose = param->b_is_trans;
    m->postops_type = param->postops_type;

    *mm = m;
    return true;
ERR:
    delete m;
    return false;
}

void fc_kernel_destroy_amx(const fc_kernel* mm) {
    if (mm) {
        delete mm;
    }
}

void fc_kernel_execute_amx(const fc_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K, size_t n_start, size_t n_end, float* dq, float* q, float* bias) {
    size_t b_d0 = K, b_d1 = N;
    if (mm->b_is_transpose) {
        b_d0 = N;
        b_d1 = K;
    }
    if (mm->i8xi8) {
        tensor2D<int8_t> a(M, K, reinterpret_cast<int8_t*>(ptr_a), lda);
        tensor2D<int8_t> b(b_d0, b_d1, reinterpret_cast<int8_t*>(ptr_b), ldb);

        if (mm->dt_c == dnnl_s8) {
            tensor2D<int8_t> c(M, N, reinterpret_cast<int8_t*>(ptr_c), ldc);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_GELU_QUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_QUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU_QUANT> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<int8_t, amx_kernel::PP::Steps::DEQUANT_BIAS_QUANT> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    ppkernel.set_q_scale(q);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == dnnl_bf16) {
            tensor2D<bfloat16> c(M, N, reinterpret_cast<bfloat16*>(ptr_c), ldc);
            if (!bias) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == dnnl_f32) {
            tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
            if (!bias) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    ppkernel.set_deq_scale(dq);
                    (*mm->i8xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        }
    } else if (mm->u8xi8) {
        tensor2D<uint8_t> a(M, K, reinterpret_cast<uint8_t*>(ptr_a), lda);
        tensor2D<int8_t> b(b_d0, b_d1, reinterpret_cast<int8_t*>(ptr_b), ldb);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->u8xi8)(a, b, n_start, n_end, pp);
    } else if (mm->bf16xbf16) {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), lda);
        tensor2D<bfloat16> b(b_d0, b_d1, reinterpret_cast<bfloat16*>(ptr_b), ldb);

        if (mm->dt_c == dnnl_bf16) {
            tensor2D<bfloat16> c(M, N, reinterpret_cast<bfloat16*>(ptr_c), ldc);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::GELU> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::NONE> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::BIAS> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == dnnl_f32) {
            tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::GELU> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> ppkernel(c);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::BIAS> ppkernel(c, bias);
                    (*mm->bf16xbf16)(a, b, n_start, n_end, ppkernel);
                }
            }
        }
    } else {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), lda);
        tensor2D<bfloat16> b(N, K, reinterpret_cast<bfloat16*>(ptr_b), ldb);

        if (mm->dt_c == dnnl_bf16) {
            tensor2D<bfloat16> c(M, N, reinterpret_cast<bfloat16*>(ptr_c), ldc);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<ov::bfloat16, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        } else if (mm->dt_c == dnnl_f32) {
            tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
            if (!(mm->postops_type & BIAS)) {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_GELU> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT> ppkernel(c);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            } else {
                if (mm->postops_type & GELU) {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS_GELU> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                } else {
                    amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::DEQUANT_BIAS> ppkernel(c, bias);
                    (*mm->bf16xi8)(a, b, n_start, n_end, ppkernel);
                }
            }
        }
    }
}

void fc_kernel_bf16w8_get_q_dq_amx(size_t K, size_t N, size_t stride, void* ptr, float* q, float* dq) {
    float min, max;
    tensor2D<bfloat16> B(K, N, reinterpret_cast<bfloat16*>(ptr), stride);
    amx_kernel::functional::get_min_max(B, min, max);
    max = std::max(std::abs(max), std::abs(min));
    *q = 127 / max;
    *dq = max / 127;
}

/// set q, dq for each fc_kernel instance
void fc_kernel_bf16w8_set_q_dq_amx(const fc_kernel* mm, float q, float dq) {
    if (!mm || !mm->bf16xi8) {
        std::cout << "fc_kernel_bf16w8_set_q_dq: created kernel is not int8 weight.\n";
        return;
    }
    mm->bf16xi8->quant_scale_B = q;
    mm->bf16xi8->dequant_scale_B = dq;
}

}