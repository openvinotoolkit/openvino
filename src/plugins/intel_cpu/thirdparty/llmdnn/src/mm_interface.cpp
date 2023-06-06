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
#include "mm_kernel_amx.hpp"
#include "utility_avx512.hpp"

namespace llmdnn {

using ov::bfloat16;
struct mm_kernel {
    std::shared_ptr<amx_kernel::Matmul<ov::bfloat16, ov::bfloat16>> bf16xbf16;
    std::shared_ptr<amx_kernel::Matmul<int8_t, int8_t>> i8xi8;
    std::shared_ptr<amx_kernel::Matmul<uint8_t, int8_t>> u8xi8;

    std::shared_ptr<amx_kernel::MatmulVector<int8_t, int8_t>> i8xi8_gemv;
    std::shared_ptr<amx_kernel::MatmulVector<ov::bfloat16, ov::bfloat16>> bf16xbf16_gemv;

    data_type_t dt_a;
    data_type_t dt_b;
    bool b_is_transpose;
};

// interface
bool mm_kernel_create(mm_kernel** mm, const mm_create_param* param) {
    mm_kernel* m = nullptr;
    if (param == nullptr || mm == nullptr) {
        std::cout << "mm_kernel_create: invalid input parameter.\n";
        goto ERR;
    }

    m = new mm_kernel;
    if (param->b_is_gemv) {
        if (param->dt_a == dnnl_s8 && param->dt_b == dnnl_s8) {
            m->i8xi8_gemv = std::make_shared<amx_kernel::MatmulVector<int8_t, int8_t>>();
        } else if (param->dt_a == dnnl_bf16 && param->dt_b == dnnl_bf16) {
            m->bf16xbf16_gemv = std::make_shared<amx_kernel::MatmulVector<bfloat16, bfloat16>>();
        } else {
            std::cout << "mm_kernel_create: unsupport gemv input type, a: " << param->dt_a << ", b: " << param->dt_b << ".\n";
            goto ERR;
        }
    } else {
        if (param->dt_a == dnnl_s8 && param->dt_b == dnnl_s8) {
            m->i8xi8 = std::make_shared<amx_kernel::Matmul<int8_t, int8_t>>(false, param->b_is_trans);
        } else if (param->dt_a == dnnl_u8 && param->dt_b == dnnl_s8) {
            m->u8xi8 = std::make_shared<amx_kernel::Matmul<uint8_t, int8_t>>(false, param->b_is_trans);
        } else if (param->dt_a == dnnl_bf16 && param->dt_b == dnnl_bf16) {
            m->bf16xbf16 = std::make_shared<amx_kernel::Matmul<bfloat16, bfloat16>>(false, param->b_is_trans);
        } else {
            std::cout << "mm_kernel_create: unsupport input type, a: " << param->dt_a << ", b: " << param->dt_b << ".\n";
            goto ERR;
        }
    }
    m->dt_a = param->dt_a;
    m->dt_b = param->dt_b;
    m->b_is_transpose = param->b_is_trans;

    *mm = m;
    return true;
ERR:
    delete m;
    return false;
}

void mm_kernel_destroy(const mm_kernel* mm) {
    if (mm) {
        delete mm;
    }
}

void mm_kernel_execute(const mm_kernel* mm, void* ptr_a, void* ptr_b, void* ptr_c, size_t lda, size_t ldb, size_t ldc,
        size_t M, size_t N, size_t K) {
    size_t b_d0 = K, b_d1 = N;
    if (mm->b_is_transpose) {
        b_d0 = N;
        b_d1 = K;
    }
    if (mm->i8xi8_gemv) {
        tensor2D<int8_t> a(M, K, reinterpret_cast<int8_t*>(ptr_a), lda);
        (*mm->i8xi8_gemv)(a, reinterpret_cast<int8_t*>(ptr_b), reinterpret_cast<int32_t*>(ptr_c));
        cvt_i32_f32(reinterpret_cast<float*>(ptr_c), reinterpret_cast<int32_t*>(ptr_c), M);
    } else if (mm->i8xi8) {
        tensor2D<int8_t> a(M, K, reinterpret_cast<int8_t*>(ptr_a), lda);
        tensor2D<int8_t> b(b_d0, b_d1, reinterpret_cast<int8_t*>(ptr_b), ldb);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->i8xi8)(a, b, 0, N, pp);
    } else if (mm->u8xi8) {
        tensor2D<uint8_t> a(M, K, reinterpret_cast<uint8_t*>(ptr_a), lda);
        tensor2D<int8_t> b(b_d0, b_d1, reinterpret_cast<int8_t*>(ptr_b), ldb);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->u8xi8)(a, b, 0, N, pp);
    } else if (mm->bf16xbf16_gemv) {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), lda);
        (*mm->bf16xbf16_gemv)(a, reinterpret_cast<bfloat16*>(ptr_b), reinterpret_cast<float*>(ptr_c));
    } else if (mm->bf16xbf16) {
        tensor2D<bfloat16> a(M, K, reinterpret_cast<bfloat16*>(ptr_a), lda);
        tensor2D<bfloat16> b(b_d0, b_d1, reinterpret_cast<bfloat16*>(ptr_b), ldb);
        tensor2D<float> c(M, N, reinterpret_cast<float*>(ptr_c), ldc);
        amx_kernel::PP::BiasGeluStore<float, amx_kernel::PP::Steps::NONE> pp(c);
        (*mm->bf16xbf16)(a, b, 0, N, pp);
    } else {
        std::cout << "mm_kernel_execute: no valid kernel created, call create first.\n";
    }
}


}