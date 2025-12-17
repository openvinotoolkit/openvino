// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.cl"

typedef struct __attribute__ ((packed)) int2x4_t { char s0; } int2x4_t;
typedef struct __attribute__ ((packed)) int2x8_t { int2x4_t s0; int2x4_t s1; } int2x8_t;
typedef struct __attribute__ ((packed)) int2x16_t { int2x4_t s0; int2x4_t s1; int2x4_t s2; int2x4_t s3; } int2x16_t;
typedef struct __attribute__ ((packed)) int2x32_t { int2x16_t s0; int2x16_t s1; } int2x32_t;

typedef struct __attribute__ ((packed)) uint2x4_t { uchar s0; } uint2x4_t;
typedef struct __attribute__ ((packed)) uint2x8_t { uint2x4_t s0; uint2x4_t s1; } uint2x8_t;
typedef struct __attribute__ ((packed)) uint2x16_t { uint2x4_t s0; uint2x4_t s1; uint2x4_t s2; uint2x4_t s3; } uint2x16_t;
typedef struct __attribute__ ((packed)) uint2x32_t { uint2x16_t s0; uint2x16_t s1; } uint2x32_t;

inline uchar4 cvt_uint2x4_to_uint8x4(uint2x4_t v) __attribute__((overloadable)) {
    uchar v0 = (v.s0 >> 0) & 0x03;
    uchar v1 = (v.s0 >> 2) & 0x03;
    uchar v2 = (v.s0 >> 4) & 0x03;
    uchar v3 = (v.s0 >> 6) & 0x03;
    return (uchar4)(v0, v1, v2, v3);
}

inline char4 cvt_int2x4_to_int8x4(int2x4_t v) __attribute__((overloadable)) {
    char r0 = (char)(v.s0 << 6) >> 6;
    char r1 = (char)(v.s0 << 4) >> 6;
    char r2 = (char)(v.s0 << 2) >> 6;
    char r3 = (char)(v.s0) >> 6;
    
    return (char4)(r0, r1, r2, r3);
}

inline uchar4 unpack_to_uchar(uint2x4_t v) __attribute__((overloadable)) {
    return cvt_uint2x4_to_uint8x4(v);
}

inline char4 unpack_to_char(int2x4_t v) __attribute__((overloadable)) {
    return cvt_int2x4_to_int8x4(v);
}

inline char4 unpack_to_char(uint2x4_t v) __attribute__((overloadable)) {
    return convert_char4(cvt_uint2x4_to_uint8x4(v));
}

// 2bit x 8 -> 8 values. Each int2x8_t has two int2x4_t.
inline uchar8 unpack_to_uchar(uint2x8_t v) __attribute__((overloadable)) {
    uchar4 v0 = unpack_to_uchar(v.s0);
    uchar4 v1 = unpack_to_uchar(v.s1);
    return (uchar8)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3);
}

inline char8 unpack_to_char(int2x8_t v) __attribute__((overloadable)) {
    char4 v0 = unpack_to_char(v.s0);
    char4 v1 = unpack_to_char(v.s1);
    return (char8)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3);
}

inline char8 unpack_to_char(uint2x8_t v) __attribute__((overloadable)) {
    char4 v0 = unpack_to_char(v.s0);
    char4 v1 = unpack_to_char(v.s1);
    return (char8)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3);
}

inline uchar16 unpack_to_uchar(uint2x16_t v) __attribute__((overloadable)) {
    uint2x8_t tmp0 = (uint2x8_t){v.s0, v.s1};
    uint2x8_t tmp1 = (uint2x8_t){v.s2, v.s3};
    uchar8 v0 = unpack_to_uchar(tmp0);
    uchar8 v1 = unpack_to_uchar(tmp1);
    return (uchar16)(v0.s0, v0.s1, v0.s2, v0.s3, v0.s4, v0.s5, v0.s6, v0.s7,
                     v1.s0, v1.s1, v1.s2, v1.s3, v1.s4, v1.s5, v1.s6, v1.s7);
}

inline float4 unpack_to_float(uint2x4_t v) __attribute__((overloadable)) {
    return convert_float4(cvt_uint2x4_to_uint8x4(v));
}

inline float4 unpack_to_float(int2x4_t v) __attribute__((overloadable)) {
    return convert_float4(cvt_int2x4_to_int8x4(v));
}

inline float8 unpack_to_float(uint2x8_t v) __attribute__((overloadable)) {
    return convert_float8(unpack_to_uchar(v));
}

inline float8 unpack_to_float(int2x8_t v) __attribute__((overloadable)) {
    return convert_float8(unpack_to_char(v));
}

#if defined(cl_khr_fp16)
inline half4 unpack_to_half(uint2x4_t v) __attribute__((overloadable)) {
    return convert_half4(cvt_uint2x4_to_uint8x4(v));
}

inline half4 unpack_to_half(int2x4_t v) __attribute__((overloadable)) {
    return convert_half4(cvt_int2x4_to_int8x4(v));
}

inline half8 unpack_to_half(uint2x8_t v) __attribute__((overloadable)) {
    return convert_half8(unpack_to_uchar(v));
}

inline half8 unpack_to_half(int2x8_t v) __attribute__((overloadable)) {
    return convert_half8(unpack_to_char(v));
}
#endif

#define UNPACK_INT2x4(target_type, value) CAT(unpack_to_, target_type)(value)
