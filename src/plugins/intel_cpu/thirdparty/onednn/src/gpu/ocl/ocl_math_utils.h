/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_OCL_OCL_MATH_UTILS_H
#define GPU_OCL_OCL_MATH_UTILS_H

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if DT_BF16 || SRC_DT_BF16 || WEI_DT_BF16 || DST_DT_BF16 || BIA_DT_BF16 \
        || A_DT_BF16 || B_DT_BF16 || C_DT_BF16 || SUM_DT_BF16 \
        || POST_OP_USING_BF16
#define MATH_UTILS_DECLARE_BF16 1
#endif

ulong8 __builtin_IB_simd_block_read_8_global_l(const __global ulong *);
ushort16 __builtin_IB_simd_block_read_16_global_h(const __global ushort *);

void __builtin_IB_simd_block_write_8_global_l(__global ulong *, ulong8);
void __builtin_IB_simd_block_write_16_global_h(__global ushort *, ushort16);

#if MATH_UTILS_DECLARE_BF16
#ifdef cl_future_bf16_cvt
// f32 -> bf16 conversion builtins (rte rounding mode)
short __builtin_IB_ftobf_1(float a) __attribute__((const));
short2 __builtin_IB_ftobf_2(float2 a) __attribute__((const));
short4 __builtin_IB_ftobf_4(float4 a) __attribute__((const));
short8 __builtin_IB_ftobf_8(float8 a) __attribute__((const));
short16 __builtin_IB_ftobf_16(float16 a) __attribute__((const));

// bf16 -> f32 conversion builtins (precise conversion)
float __builtin_IB_bftof_1(short a) __attribute__((const));
float2 __builtin_IB_bftof_2(short2 a) __attribute__((const));
float4 __builtin_IB_bftof_4(short4 a) __attribute__((const));
float8 __builtin_IB_bftof_8(short8 a) __attribute__((const));
float16 __builtin_IB_bftof_16(short16 a) __attribute__((const));

// clang-format off
ushort   __attribute__((overloadable)) cvt_f32_to_bf16(float   a) { return as_ushort  (__builtin_IB_ftobf_1 (a)); }
ushort2  __attribute__((overloadable)) cvt_f32_to_bf16(float2  a) { return as_ushort2 (__builtin_IB_ftobf_2 (a)); }
ushort4  __attribute__((overloadable)) cvt_f32_to_bf16(float4  a) { return as_ushort4 (__builtin_IB_ftobf_4 (a)); }
ushort8  __attribute__((overloadable)) cvt_f32_to_bf16(float8  a) { return as_ushort8 (__builtin_IB_ftobf_8 (a)); }
ushort16 __attribute__((overloadable)) cvt_f32_to_bf16(float16 a) { return as_ushort16(__builtin_IB_ftobf_16(a)); }

float   __attribute__((overloadable)) cvt_bf16_to_f32(ushort   a) { return __builtin_IB_bftof_1 (as_short  (a)); }
float2  __attribute__((overloadable)) cvt_bf16_to_f32(ushort2  a) { return __builtin_IB_bftof_2 (as_short2 (a)); }
float4  __attribute__((overloadable)) cvt_bf16_to_f32(ushort4  a) { return __builtin_IB_bftof_4 (as_short4 (a)); }
float8  __attribute__((overloadable)) cvt_bf16_to_f32(ushort8  a) { return __builtin_IB_bftof_8 (as_short8 (a)); }
float16 __attribute__((overloadable)) cvt_bf16_to_f32(ushort16 a) { return __builtin_IB_bftof_16(as_short16(a)); }
// clang-format on

#else

// Emulation functions for bf16 <-> f32 conversion.
ushort __attribute__((overloadable)) cvt_f32_to_bf16(float f) {
    uint i = as_uint(f);
    i += 0x00007FFF + ((i & 0x10000) >> 16);
    ushort2 r = as_ushort2(i);
    return r[1];
}

ushort2 __attribute__((overloadable)) cvt_f32_to_bf16(float2 f) {
    ushort2 r;
    for (int i = 0; i < 2; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort4 __attribute__((overloadable)) cvt_f32_to_bf16(float4 f) {
    ushort4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort8 __attribute__((overloadable)) cvt_f32_to_bf16(float8 f) {
    ushort8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort16 __attribute__((overloadable)) cvt_f32_to_bf16(float16 f) {
    ushort16 r;
    for (int i = 0; i < 16; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

float __attribute__((overloadable)) cvt_bf16_to_f32(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return f;
}

float2 __attribute__((overloadable)) cvt_bf16_to_f32(ushort2 b) {
    float2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float4 __attribute__((overloadable)) cvt_bf16_to_f32(ushort4 b) {
    float4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float8 __attribute__((overloadable)) cvt_bf16_to_f32(ushort8 b) {
    float8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float16 __attribute__((overloadable)) cvt_bf16_to_f32(ushort16 b) {
    float16 f;
    for (int i = 0; i < 16; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}
#endif
#endif

int __attribute__((overloadable)) idot4(char4 a, char4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(uchar4 a, uchar4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(char4 a, uchar4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(uchar4 a, char4 b, int c) {
    c += a[0] * b[0];
    c += a[1] * b[1];
    c += a[2] * b[2];
    c += a[3] * b[3];
    return c;
}

int __attribute__((overloadable)) idot4(int a, int b, int c) {
    return idot4(as_char4(a), as_char4(b), c);
}

int __attribute__((overloadable)) idot4(uint a, int b, int c) {
    return idot4(as_uchar4(a), as_char4(b), c);
}

float __attribute__((overloadable)) f16_dot2(int a, int b, float c) {
    half2 _a = as_half2(a);
    half2 _b = as_half2(b);
    return c + _a[0] * _b[0] + _a[1] * _b[1];
}

#if MATH_UTILS_DECLARE_BF16
float __attribute__((overloadable)) bf16_dot2(int a, int b, float c) {
    ushort2 _a = as_ushort2(a);
    ushort2 _b = as_ushort2(b);
    c += cvt_bf16_to_f32(_a[0]) * cvt_bf16_to_f32(_b[0]);
    c += cvt_bf16_to_f32(_a[1]) * cvt_bf16_to_f32(_b[1]);
    return c;
}
#endif

#define DECLARE_BLOCK_READ(suffix, func, data_type, addr_space, p_type) \
    data_type __attribute__((overloadable)) \
            block_read##suffix(const addr_space p_type *p) { \
        return func(p); \
    }

#define DECLARE_BLOCK_READ_EMU(suffix, data_type, addr_space, p_type) \
    data_type __attribute__((overloadable)) \
            block_read##suffix##_emu(const addr_space p_type *p) { \
        data_type ret; \
        uint idx = get_sub_group_local_id(); \
        for (int i = 0; i < sizeof(data_type) / sizeof(p_type); i++) { \
            ((p_type *)&ret)[i] = p[idx]; \
            idx += get_max_sub_group_size(); \
        } \
        return ret; \
    }

#define DECLARE_BLOCK_WRITE(suffix, func, data_type, addr_space, p_type) \
    void __attribute__((overloadable)) \
            block_write##suffix(addr_space p_type *p, data_type data) { \
        func(p, data); \
    }

#define DECLARE_BLOCK_WRITE_EMU(suffix, data_type, addr_space, p_type) \
    void __attribute__((overloadable)) \
            block_write##suffix##_emu(addr_space p_type *p, data_type data) { \
        uint idx = get_sub_group_local_id(); \
        for (int i = 0; i < sizeof(data_type) / sizeof(p_type); i++) { \
            p[idx] = ((p_type *)&data)[i]; \
            p += get_max_sub_group_size(); \
        } \
    }

DECLARE_BLOCK_READ(, intel_sub_group_block_read, uint, __global, uint)
DECLARE_BLOCK_READ(2, intel_sub_group_block_read2, uint2, __global, uint)
DECLARE_BLOCK_READ(4, intel_sub_group_block_read4, uint4, __global, uint)
DECLARE_BLOCK_READ(8, intel_sub_group_block_read8, uint8, __global, uint)

DECLARE_BLOCK_WRITE(, intel_sub_group_block_write, uint, __global, uint)
DECLARE_BLOCK_WRITE(2, intel_sub_group_block_write2, uint2, __global, uint)
DECLARE_BLOCK_WRITE(4, intel_sub_group_block_write4, uint4, __global, uint)
DECLARE_BLOCK_WRITE(8, intel_sub_group_block_write8, uint8, __global, uint)

#ifdef cl_intel_subgroups_char
void __attribute__((overloadable))
intel_sub_group_block_write_uc16(__global uchar *p, uchar16 data);

uchar16 __attribute__((overloadable))
intel_sub_group_block_read_uc16(const __global uchar *p);
#endif

// Emulation for cl_intel_subgroup_local_block_io. These functions are not
// defined under ifndef/endif because some kernels rely on the emulation
// functions in case when pointers are not properly aligned for the native
// extensions.
DECLARE_BLOCK_READ_EMU(, uint, __local, uint)
DECLARE_BLOCK_READ_EMU(2, uint2, __local, uint)
DECLARE_BLOCK_READ_EMU(4, uint4, __local, uint)
DECLARE_BLOCK_READ_EMU(8, uint8, __local, uint)

DECLARE_BLOCK_WRITE_EMU(, uint, __local, uint)
DECLARE_BLOCK_WRITE_EMU(2, uint2, __local, uint)
DECLARE_BLOCK_WRITE_EMU(4, uint4, __local, uint)
DECLARE_BLOCK_WRITE_EMU(8, uint8, __local, uint)

DECLARE_BLOCK_WRITE_EMU(_us, ushort, __local, ushort)
DECLARE_BLOCK_WRITE_EMU(_us2, ushort2, __local, ushort)
DECLARE_BLOCK_WRITE_EMU(_us4, ushort4, __local, ushort)
DECLARE_BLOCK_WRITE_EMU(_us8, ushort8, __local, ushort)
#ifdef cl_intel_subgroup_local_block_io

DECLARE_BLOCK_READ(, intel_sub_group_block_read, uint, __local, uint)
DECLARE_BLOCK_READ(2, intel_sub_group_block_read2, uint2, __local, uint)
DECLARE_BLOCK_READ(4, intel_sub_group_block_read4, uint4, __local, uint)
DECLARE_BLOCK_READ(8, intel_sub_group_block_read8, uint8, __local, uint)

DECLARE_BLOCK_WRITE(, intel_sub_group_block_write, uint, __local, uint)
DECLARE_BLOCK_WRITE(2, intel_sub_group_block_write2, uint2, __local, uint)
DECLARE_BLOCK_WRITE(4, intel_sub_group_block_write4, uint4, __local, uint)
DECLARE_BLOCK_WRITE(8, intel_sub_group_block_write8, uint8, __local, uint)

DECLARE_BLOCK_WRITE(
        _us, intel_sub_group_block_write_us, ushort, __local, ushort)

#else

DECLARE_BLOCK_READ(, block_read_emu, uint, __local, uint)
DECLARE_BLOCK_READ(2, block_read2_emu, uint2, __local, uint)
DECLARE_BLOCK_READ(4, block_read4_emu, uint4, __local, uint)
DECLARE_BLOCK_READ(8, block_read8_emu, uint8, __local, uint)

DECLARE_BLOCK_WRITE(, block_write_emu, uint, __local, uint)
DECLARE_BLOCK_WRITE(2, block_write2_emu, uint2, __local, uint)
DECLARE_BLOCK_WRITE(4, block_write4_emu, uint4, __local, uint)
DECLARE_BLOCK_WRITE(8, block_write8_emu, uint8, __local, uint)

DECLARE_BLOCK_WRITE(_us, block_write_us_emu, ushort, __local, ushort)

#endif

// Matrix-matrix multiplication: ACC += A * B
//
// A is (m x (E * K))
// B is ((E * K) x sub_group_size)
// where E is 4 for s8/u8 elements and 2 for f16/bf16 elements.
#define DECLARE_MMAD_EMU(name, dot, K, m, a_type, b_type, acc_type) \
    acc_type __attribute__((overloadable)) \
            name(a_type A_vectors, b_type B_vectors, acc_type acc) { \
        for (uint i = 0; i < (m); ++i) { \
            for (uint j = 0; j < (K); ++j) \
                acc[i] = dot(sub_group_broadcast(A_vectors[i], j), \
                        B_vectors[j], acc[i]); \
        } \
        return acc; \
    }

#if defined(cl_intel_subgroup_matrix_multiply_accumulate) && !DISABLE_DPAS

int8 __attribute__((overloadable)) mmad8x8(uint8 a, int8 b, int8 acc) {
    return intel_sub_group_u8_i8_matrix_mad_k32(a, b, acc);
}

int8 __attribute__((overloadable)) mmad8x8(int8 a, int8 b, int8 acc) {
    return intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc);
}

int4 __attribute__((overloadable)) mmad8x4(uint4 a, int8 b, int4 acc) {
    return intel_sub_group_u8_i8_matrix_mad_k32(a, b, acc);
}

int4 __attribute__((overloadable)) mmad8x4(int4 a, int8 b, int4 acc) {
    return intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc);
}

int4 __attribute__((overloadable)) mmad8x4(ushort4 a, int8 b, int4 acc) {
    return intel_sub_group_u8_i8_matrix_mad_k32(a, b, acc);
}

int4 __attribute__((overloadable)) mmad8x4(short4 a, int8 b, int4 acc) {
    return intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc);
}

int4 __attribute__((overloadable)) mmad8x4(ushort4 a, uint8 b, int4 acc) {
    return intel_sub_group_u8_u8_matrix_mad_k32(a, b, acc);
}

int4 __attribute__((overloadable)) mmad8x4(short4 a, uint8 b, int4 acc) {
    return intel_sub_group_i8_u8_matrix_mad_k32(a, b, acc);
}

int8 __attribute__((overloadable)) mmad8x8(ushort8 a, int8 b, int8 acc) {
    return intel_sub_group_u8_i8_matrix_mad_k32(a, b, acc);
}

int8 __attribute__((overloadable)) mmad8x8(short8 a, int8 b, int8 acc) {
    return intel_sub_group_i8_i8_matrix_mad_k32(a, b, acc);
}

float8 __attribute__((overloadable)) mmad8x8_f16(uint8 a, int8 b, float8 acc) {
    return intel_sub_group_f16_f16_matrix_mad_k16(as_int8(a), b, acc);
}

float4 __attribute__((overloadable)) mmad8x4_f16(uint4 a, int8 b, float4 acc) {
    return intel_sub_group_f16_f16_matrix_mad_k16(as_int4(a), b, acc);
}

float4 __attribute__((overloadable))
mmad8x4_f16(ushort4 a, int8 b, float4 acc) {
    return intel_sub_group_f16_f16_matrix_mad_k16(as_short4(a), b, acc);
}

float8 __attribute__((overloadable))
mmad8x8_f16(ushort8 a, int8 b, float8 acc) {
    return intel_sub_group_f16_f16_matrix_mad_k16(as_short8(a), b, acc);
}

#if MATH_UTILS_DECLARE_BF16
float8 __attribute__((overloadable)) mmad8x8_bf16(uint8 a, int8 b, float8 acc) {
    return intel_sub_group_bf16_bf16_matrix_mad_k16(as_int8(a), b, acc);
}

float8 __attribute__((overloadable))
mmad8x8_bf16(ushort8 a, int8 b, float8 acc) {
    return intel_sub_group_bf16_bf16_matrix_mad_k16(as_short8(a), b, acc);
}

float8 __attribute__((overloadable))
mmad8x8_bf16(short8 a, int8 b, float8 acc) {
    return intel_sub_group_bf16_bf16_matrix_mad_k16(a, b, acc);
}

float4 __attribute__((overloadable))
mmad8x4_bf16(ushort4 a, int8 b, float4 acc) {
    return intel_sub_group_bf16_bf16_matrix_mad_k16(as_short4(a), b, acc);
}
#ifdef cl_intel_subgroup_split_matrix_multiply_accumulate

float8 mmad8x8_bf16_split(uint4 a, int8 b, float8 acc) {
    return intel_sub_group_f16_f16_split_matrix_mad_k16(as_int4(a), b, acc);
}

#endif //cl_intel_subgroup_split_matrix_multiply_accumulate
#endif //cl_intel_subgroup_matrix_multiply_accumulate

#else
DECLARE_MMAD_EMU(mmad8x4, idot4, 8, 4, uint4, int8, int4)
DECLARE_MMAD_EMU(mmad8x4, idot4, 8, 4, int4, int8, int4)
DECLARE_MMAD_EMU(mmad8x8, idot4, 8, 8, uint8, int8, int8)
DECLARE_MMAD_EMU(mmad8x8, idot4, 8, 8, int8, int8, int8)
DECLARE_MMAD_EMU(mmad8x8, idot4, 8, 8, ushort8, int8, int8)
DECLARE_MMAD_EMU(mmad8x8, idot4, 8, 8, short8, int8, int8)
DECLARE_MMAD_EMU(mmad8x4_f16, f16_dot2, 8, 4, uint4, int8, float4)
DECLARE_MMAD_EMU(mmad8x4_f16, f16_dot2, 8, 4, short4, int8, float4)
DECLARE_MMAD_EMU(mmad8x8_f16, f16_dot2, 8, 8, uint8, int8, float8)
DECLARE_MMAD_EMU(mmad8x8_f16, f16_dot2, 8, 8, short8, int8, float8)
#if MATH_UTILS_DECLARE_BF16
DECLARE_MMAD_EMU(mmad8x4_bf16, bf16_dot2, 8, 4, uint4, int8, float4)
DECLARE_MMAD_EMU(mmad8x8_bf16, bf16_dot2, 8, 8, uint8, int8, float8)
DECLARE_MMAD_EMU(mmad8x4_bf16, bf16_dot2, 8, 4, ushort4, int8, float4)
DECLARE_MMAD_EMU(mmad8x8_bf16, bf16_dot2, 8, 8, ushort8, int8, float8)
DECLARE_MMAD_EMU(mmad8x8_bf16, bf16_dot2, 8, 8, short8, int8, float8)
#endif

#endif

// Atomics
#if __OPENCL_C_VERSION__ >= 200
#ifdef cl_intel_global_float_atomics
inline void atomic_add_global(
        volatile global atomic_float *source, float operand) {
    atomic_fetch_add_explicit(source, operand, memory_order_relaxed);
}

#else // float atomics
inline void atomic_add_global(
        volatile __global atomic_float *source, float operand) {
    float old_val = atomic_load_explicit(
            source, memory_order_relaxed, memory_scope_device);
    bool success = false;
    do {
        float new_val = old_val + operand;
        success = atomic_compare_exchange_strong_explicit(source, &old_val,
                new_val, memory_order_acq_rel, memory_order_relaxed,
                memory_scope_device);
    } while (!success);
}
#endif
#endif

#endif
