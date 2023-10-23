//  Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

typedef struct __attribute__ ((packed)) int4x2_t { char s0; } int4x2_t;
typedef struct __attribute__ ((packed)) int4x4_t { int4x2_t s0; int4x2_t s1; } int4x4_t;
typedef struct __attribute__ ((packed)) int4x8_t { int4x2_t s0; int4x2_t s1; int4x2_t s2; int4x2_t s3; } int4x8_t;
typedef struct __attribute__ ((packed)) int4x16_t { int4x2_t s0; int4x2_t s1; int4x2_t s2; int4x2_t s3; int4x2_t s4; int4x2_t s5; int4x2_t s6; int4x2_t s7; } int4x16_t;

typedef struct __attribute__ ((packed)) uint4x2_t { uchar s0; } uint4x2_t;
typedef struct __attribute__ ((packed)) uint4x4_t { uint4x2_t s0; uint4x2_t s1; } uint4x4_t;
typedef struct __attribute__ ((packed)) uint4x8_t { uint4x2_t s0; uint4x2_t s1; uint4x2_t s2; uint4x2_t s3; } uint4x8_t;
typedef struct __attribute__ ((packed)) uint4x16_t { uint4x2_t s0; uint4x2_t s1; uint4x2_t s2; uint4x2_t s3; uint4x2_t s4; uint4x2_t s5; uint4x2_t s6; uint4x2_t s7; } uint4x16_t;

inline uchar2 cvt_uint4x2_to_uint8x2(uint4x2_t v) __attribute__((overloadable)) {
    const uchar v0 = v.s0 & 0x0F;
    const uchar v1 = (v.s0 & 0xF0) >> 4;
    return (uchar2)(v0, v1);
}

inline char2 cvt_int4x2_to_int8x2(int4x2_t v) __attribute__((overloadable)) {
    const char s_bit = (v.s0 & convert_char(0x08));
    const char mask = s_bit > 0 ? convert_char(0xF0) : convert_char(0x00);
    const char v0 = (v.s0 & convert_char(0x0F)) | mask;
    const char v1 = v.s0 >> 4;
    return (char2)(v0, v1);
}

inline half2 unpack_to_half(uint4x2_t v) __attribute__((overloadable)) {
    return convert_half2(cvt_uint4x2_to_uint8x2(v));
}

inline float2 unpack_to_float(uint4x2_t v) __attribute__((overloadable)) {
    return convert_float2(cvt_uint4x2_to_uint8x2(v));
}

inline half2 unpack_to_half(int4x2_t v) __attribute__((overloadable)) {
    return convert_half2(cvt_int4x2_to_int8x2(v));
}

inline float2 unpack_to_float(int4x2_t v) __attribute__((overloadable)) {
    return convert_float2(cvt_int4x2_to_int8x2(v));
}

inline half4 unpack_to_half(uint4x4_t v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline float4 unpack_to_float(uint4x4_t v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half4 unpack_to_half(int4x4_t v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline float4 unpack_to_float(int4x4_t v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half8 unpack_to_half(uint4x8_t v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    half2 f2 = unpack_to_half(v.s2);
    half2 f3 = unpack_to_half(v.s3);
    return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline float8 unpack_to_float(uint4x8_t v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    float2 f2 = unpack_to_float(v.s2);
    float2 f3 = unpack_to_float(v.s3);
    return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline half8 unpack_to_half(int4x8_t v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    half2 f2 = unpack_to_half(v.s2);
    half2 f3 = unpack_to_half(v.s3);
    return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline float8 unpack_to_float(int4x8_t v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    float2 f2 = unpack_to_float(v.s2);
    float2 f3 = unpack_to_float(v.s3);
    return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

#define UNPACK_INT4x2(target_type, value) CAT(unpack_to_, target_type)(value)
