//  Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

inline uchar2 cvt_uint4x2_to_uint8x2(uchar v) __attribute__((overloadable)) {
    const uchar v0 = v & 0x0F;
    const uchar v1 = (v & 0xF0) >> 4;
    return (uchar2)(v0, v1);
}

inline char2 cvt_int4x2_to_int8x2(char v) __attribute__((overloadable)) {
    const char s_bit = (v & convert_char(0x08));
    const char mask = s_bit > 0 ? convert_char(0xF0) : convert_char(0x00);
    const char v0 = (v & convert_char(0x0F)) | mask;
    const char v1 = v >> 4;
    return (char2)(v0, v1);
}

inline half2 unpack_to_half(uchar v) __attribute__((overloadable)) {
    return convert_half2(cvt_uint4x2_to_uint8x2(v));
}

inline float2 unpack_to_float(uchar v) __attribute__((overloadable)) {
    return convert_float2(cvt_uint4x2_to_uint8x2(v));
}

inline half2 unpack_to_half(char v) __attribute__((overloadable)) {
    return convert_half2(cvt_int4x2_to_int8x2(v));
}

inline float2 unpack_to_float(char v) __attribute__((overloadable)) {
    return convert_float2(cvt_int4x2_to_int8x2(v));
}

inline half4 unpack_to_half(uchar2 v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline float4 unpack_to_float(uchar2 v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half4 unpack_to_half(char2 v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    return (half4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline float4 unpack_to_float(char2 v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    return (float4)(f0.s0, f0.s1, f1.s0, f1.s1);
}

inline half8 unpack_to_half(uchar4 v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    half2 f2 = unpack_to_half(v.s2);
    half2 f3 = unpack_to_half(v.s3);
    return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline float8 unpack_to_float(uchar4 v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    float2 f2 = unpack_to_float(v.s2);
    float2 f3 = unpack_to_float(v.s3);
    return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline half8 unpack_to_half(char4 v) __attribute__((overloadable)) {
    half2 f0 = unpack_to_half(v.s0);
    half2 f1 = unpack_to_half(v.s1);
    half2 f2 = unpack_to_half(v.s2);
    half2 f3 = unpack_to_half(v.s3);
    return (half8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

inline float8 unpack_to_float(char4 v) __attribute__((overloadable)) {
    float2 f0 = unpack_to_float(v.s0);
    float2 f1 = unpack_to_float(v.s1);
    float2 f2 = unpack_to_float(v.s2);
    float2 f3 = unpack_to_float(v.s3);
    return (float8)(f0.s0, f0.s1, f1.s0, f1.s1, f2.s0, f2.s1, f3.s0, f3.s1);
}

#define UNPACK_INT4x2(target_type, value) CAT(unpack_to_, target_type)(value)
