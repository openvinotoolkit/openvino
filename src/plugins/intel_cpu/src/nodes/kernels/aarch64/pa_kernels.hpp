// Copyright (C) 2024 FUJITSU LIMITED
// SPDX-License-Identifier: Apache-2.0
//
#include <cstddef>

#if defined(HAVE_SVE)
#include "arm_sve.h"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

// #define prefetch_bytes(bytes, sel, advance, src)

template<typename TA, typename TB>
void cvt_copy(TA* dst, TB* src, size_t n) {
    size_t i = 0;
    for ( ; i < n; i++) {
        dst[i] = src[i];
    }
}

template<>
void cvt_copy<float, float>(float* dst, float* src, size_t n) {
    size_t i = 0;
    dst = reinterpret_cast<float32_t*>(dst);
    src = reinterpret_cast<float32_t*>(src);
    auto sve_pg = svptrue_b32();
    for ( ; i + svcntw() <= n; i += svcntw()) {
        svfloat32_t vb = svld1_f32(sve_pg, src + i);
        svst1_f32(sve_pg, dst + i, vb);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

template<typename T>
static void attn_acc_value_block(float* out, float* weight, T* v, size_t S, size_t block_size) {
    for (size_t j = 0; j < block_size; j++) {
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * v[i];
        }
        v += S;
    }
}

static void attn_acc_value_block(float* out, float* weight, uint8_t* v, size_t S, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto sve_pg = svptrue_b32();
    size_t j = 0;
    for (; j < block_size; ++j) {
        auto v0 = reinterpret_cast<float*>(v);
        v = v + 8;
        svfloat32_t attn_w_vec0 = svdup_n_f32(weight[0]);
        svfloat32_t zp = svdup_n_f32(v0[1]);
        svfloat32_t sc = svdup_n_f32(v0[0]);
        size_t i = 0;
        for (; i + svcntw() < S; i+=svcntw()) {
            auto v_out = svld1_f32(sve_pg, out + i);
            svuint32_t reg1  = svld1ub_u32(sve_pg, v + i);
            svfloat32_t reg2 = svcvt_f32_u32_z(sve_pg, reg1);
            svfloat32_t reg3 = svsub_f32_z(sve_pg, reg2, zp);
            svfloat32_t reg4 = svmul_f32_z(sve_pg, reg3, sc);
            v_out = svmla_f32_x(sve_pg, v_out, attn_w_vec0, reg4);
            svst1_f32(sve_pg, out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * (v[i] - v0[1]) * v0[0];
        }
        v += S;
        weight += 1;
    }
    return;
}

template<typename TA, typename TB>
static void dot_product_block(TA* a, TB* b, float* c, size_t n, size_t block_size) {
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
}

template<typename TA>
static void dot_product_block(TA* a, uint8_t* b, float* c, size_t n, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        auto b0 = reinterpret_cast<float*>(b);
        b += 8;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
}

static void dot_product_block(float* a, uint8_t* b, float* c, size_t n, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto sve_pg = svptrue_b32();
    size_t j = 0;
    a = reinterpret_cast<float*>(a);
    for (; j < block_size; j++) {
        svfloat32_t vsum = svdup_n_f32(0.0f);
        auto b0 = reinterpret_cast<float*>(b);
        auto v_zp = svdup_n_f32(b0[1]);
        size_t i = 0;
        b += 8;
        for (; i + svcntw() < n; i+=svcntw()) {
            auto va = svld1_f32(sve_pg, a + i);
            svuint32_t reg1  = svld1ub_u32(sve_pg, b + i);
            svfloat32_t reg2 = svcvt_f32_u32_z(sve_pg, reg1);
            svfloat32_t vb = svsub_f32_z(sve_pg, reg2, v_zp);
            vsum = svmla_f32_m(sve_pg, vsum, va, vb);
        }
        float32_t sum = svaddv_f32(sve_pg, vsum);
        for (; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
    return;
}

template<typename T>
static void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
    for ( ; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}

static void attn_reduce(float* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
    auto sve_pg = svptrue_b32();
    for ( ; i + svcntw() < S; i+=svcntw()) {
        auto* src = temp + i;
        auto result_vec_fp32 = svdup_n_f32(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = svld1_f32(sve_pg, src);
            result_vec_fp32 = svadd_f32_z(sve_pg, result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        svst1_f32(sve_pg, dst + i, result_vec_fp32);
    }
    for ( ; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}


}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov

#endif