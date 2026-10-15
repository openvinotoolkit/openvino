//  Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ================================================================================================
// U2 (2-bit unsigned) support - 4 values per byte
// ================================================================================================

#include "common.cl"

typedef struct __attribute__ ((packed)) uint2x4_t { uchar s0; } uint2x4_t;
typedef struct __attribute__ ((packed)) uint2x8_t { uint2x4_t s0; uint2x4_t s1; } uint2x8_t;
typedef struct __attribute__ ((packed)) uint2x16_t { uint2x4_t s0; uint2x4_t s1; uint2x4_t s2; uint2x4_t s3; } uint2x16_t;

// Unpack 4 U2 values from 1 byte
inline uchar4 cvt_uint2x4_to_uint8x4(uint2x4_t v) __attribute__((overloadable)) {
    const uchar v0 = v.s0 & 0x03;           // bits [1:0]
    const uchar v1 = (v.s0 >> 2) & 0x03;    // bits [3:2]
    const uchar v2 = (v.s0 >> 4) & 0x03;    // bits [5:4]
    const uchar v3 = (v.s0 >> 6) & 0x03;    // bits [7:6]
    return (uchar4)(v0, v1, v2, v3);
}

// Pack 4 uchar values into 1 U2 byte
inline uchar cvt_uint8x4_to_uint2x4(uchar4 v) __attribute__((overloadable)) {
    uchar result = 0;
    result |= (v.s0 & 0x03);
    result |= (v.s1 & 0x03) << 2;
    result |= (v.s2 & 0x03) << 4;
    result |= (v.s3 & 0x03) << 6;
    return result;
}

// Overloads for U2x4 (4 values)
inline uchar4 unpack_to_uchar(uint2x4_t v) __attribute__((overloadable)) {
    return cvt_uint2x4_to_uint8x4(v);
}

inline char4 unpack_to_char(uint2x4_t v) __attribute__((overloadable)) {
    return convert_char4(cvt_uint2x4_to_uint8x4(v));
}

// Overloads for U2x8 (8 values from 2 bytes)
inline uchar8 unpack_to_uchar(uint2x8_t v) __attribute__((overloadable)) {
    uchar4 v0 = unpack_to_uchar(v.s0);
    uchar4 v1 = unpack_to_uchar(v.s1);
    return (uchar8)(v0.s0, v0.s1, v0.s2, v0.s3, v1.s0, v1.s1, v1.s2, v1.s3);
}

inline char8 unpack_to_char(uint2x8_t v) __attribute__((overloadable)) {
    return convert_char8(unpack_to_uchar(v));
}

// Overloads for U2x16 (16 values from 4 bytes)
inline uchar16 unpack_to_uchar(uint2x16_t v) __attribute__((overloadable)) {
    uchar4 v0 = unpack_to_uchar(v.s0);
    uchar4 v1 = unpack_to_uchar(v.s1);
    uchar4 v2 = unpack_to_uchar(v.s2);
    uchar4 v3 = unpack_to_uchar(v.s3);
    return (uchar16)(v0.s0, v0.s1, v0.s2, v0.s3, 
                     v1.s0, v1.s1, v1.s2, v1.s3,
                     v2.s0, v2.s1, v2.s2, v2.s3,
                     v3.s0, v3.s1, v3.s2, v3.s3);
}

inline char16 unpack_to_char(uint2x16_t v) __attribute__((overloadable)) {
    return convert_char16(unpack_to_uchar(v));
}

// Float unpacking for U2x4
inline float4 unpack_to_float(uint2x4_t v) __attribute__((overloadable)) {
    return convert_float4(cvt_uint2x4_to_uint8x4(v));
}

// Float unpacking for U2x8
inline float8 unpack_to_float(uint2x8_t v) __attribute__((overloadable)) {
    float4 f0 = unpack_to_float(v.s0);
    float4 f1 = unpack_to_float(v.s1);
    return (float8)(f0.s0, f0.s1, f0.s2, f0.s3, f1.s0, f1.s1, f1.s2, f1.s3);
}

#if defined(cl_khr_fp16)
// Half unpacking for U2x4
inline half4 unpack_to_half(uint2x4_t v) __attribute__((overloadable)) {
    return convert_half4(cvt_uint2x4_to_uint8x4(v));
}

// Half unpacking for U2x8
inline half8 unpack_to_half(uint2x8_t v) __attribute__((overloadable)) {
    half4 f0 = unpack_to_half(v.s0);
    half4 f1 = unpack_to_half(v.s1);
    return (half8)(f0.s0, f0.s1, f0.s2, f0.s3, f1.s0, f1.s1, f1.s2, f1.s3);
}
#endif

// Macros for generic unpacking
#define UNPACK_UINT2x4(target_type, v) CAT(unpack_to_, target_type)(v)
#define UNPACK_UINT2x8(target_type, v) CAT(unpack_to_, target_type)(v)
#define UNPACK_UINT2x16(target_type, v) CAT(unpack_to_, target_type)(v)
