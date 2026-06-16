// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common.cl"

// The bfloat16 conversion built-ins are provided by the cl_intel_bfloat16_conversions
// extension. The compiler defines the macro named after the EXTENSION (not after the
// built-in functions), so the availability must be probed via cl_intel_bfloat16_conversions.
// When present we enable the extension and map the helpers onto the hardware built-ins;
// otherwise a portable software emulation is used.
#ifdef cl_intel_bfloat16_conversions
#pragma OPENCL EXTENSION cl_intel_bfloat16_conversions : enable

// ===================== Hardware built-in bfloat16 conversions =====================

// --- scalar ---
#define _convert_as_bfloat16_float(val) intel_convert_as_bfloat16_float(val)

inline ushort __attribute__((overloadable)) _convert_bfloat16_as_ushort(float val)
{
    return intel_convert_bfloat16_as_ushort(val);
}

// --- bfloat16_as_ushort (float -> ushort) vectorized ---
inline ushort2 __attribute__((overloadable)) _convert_bfloat162_as_ushort2(float2 val)
{
    return intel_convert_bfloat162_as_ushort2(val);
}

inline ushort3 __attribute__((overloadable)) _convert_bfloat163_as_ushort3(float3 val)
{
    return intel_convert_bfloat163_as_ushort3(val);
}

inline ushort4 __attribute__((overloadable)) _convert_bfloat164_as_ushort4(float4 val)
{
    return intel_convert_bfloat164_as_ushort4(val);
}

inline ushort8 __attribute__((overloadable)) _convert_bfloat168_as_ushort8(float8 val)
{
    return intel_convert_bfloat168_as_ushort8(val);
}

inline ushort16 __attribute__((overloadable)) _convert_bfloat1616_as_ushort16(float16 val)
{
    return intel_convert_bfloat1616_as_ushort16(val);
}

// --- as_bfloat16_float (ushort -> float) vectorized ---
#define _convert_as_bfloat162_float2(val)   intel_convert_as_bfloat162_float2(val)
#define _convert_as_bfloat163_float3(val)   intel_convert_as_bfloat163_float3(val)
#define _convert_as_bfloat164_float4(val)   intel_convert_as_bfloat164_float4(val)
#define _convert_as_bfloat168_float8(val)   intel_convert_as_bfloat168_float8(val)
#define _convert_as_bfloat1616_float16(val) intel_convert_as_bfloat1616_float16(val)

#else

// ===================== Software fallback bfloat16 conversions =====================

// --- scalar ---
inline float _convert_as_bfloat16_float(ushort source) {
    uint u = 0;
    //sign
    if ( (source>>15) ) { 
        u = 1 << 31;
    }
    //exponent
    u += ( ( (source >> 7) & 0b11111111)) << 23;
    //fraction 
    u += (source & 0b1111111) << 16;
    float* f = (float*)&u;
    return *f;
}

inline ushort __attribute__((overloadable)) _convert_bfloat16_as_ushort(float source) {
    // float -> bfloat16 using round-to-nearest, ties-to-even (ROUND_MODE_TO_NEAREST_EVEN).
    // Mirrors ov::bfloat16::round_to_nearest_even in
    // src/core/include/openvino/core/type/bfloat16.hpp.
    uint u = as_uint(source);
    return (ushort)((u + ((u & 0x00010000u) >> 1)) >> 16);
}

// --- bfloat16_as_ushort (float -> ushort) vectorized ---
inline ushort2 __attribute__((overloadable)) _convert_bfloat162_as_ushort2(float2 source) {
    return (ushort2)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1));
}

inline ushort3 __attribute__((overloadable)) _convert_bfloat163_as_ushort3(float3 source) {
    return (ushort3)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1),
                     _convert_bfloat16_as_ushort(source.s2));
}

inline ushort4 __attribute__((overloadable)) _convert_bfloat164_as_ushort4(float4 source) {
    return (ushort4)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1),
                     _convert_bfloat16_as_ushort(source.s2),
                     _convert_bfloat16_as_ushort(source.s3));
}

inline ushort8 __attribute__((overloadable)) _convert_bfloat168_as_ushort8(float8 source) {
    return (ushort8)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1),
                     _convert_bfloat16_as_ushort(source.s2),
                     _convert_bfloat16_as_ushort(source.s3),
                     _convert_bfloat16_as_ushort(source.s4),
                     _convert_bfloat16_as_ushort(source.s5),
                     _convert_bfloat16_as_ushort(source.s6),
                     _convert_bfloat16_as_ushort(source.s7));
}

inline ushort16 __attribute__((overloadable)) _convert_bfloat1616_as_ushort16(float16 source) {
    return (ushort16)(_convert_bfloat16_as_ushort(source.s0),
                      _convert_bfloat16_as_ushort(source.s1),
                      _convert_bfloat16_as_ushort(source.s2),
                      _convert_bfloat16_as_ushort(source.s3),
                      _convert_bfloat16_as_ushort(source.s4),
                      _convert_bfloat16_as_ushort(source.s5),
                      _convert_bfloat16_as_ushort(source.s6),
                      _convert_bfloat16_as_ushort(source.s7),
                      _convert_bfloat16_as_ushort(source.s8),
                      _convert_bfloat16_as_ushort(source.s9),
                      _convert_bfloat16_as_ushort(source.sa),
                      _convert_bfloat16_as_ushort(source.sb),
                      _convert_bfloat16_as_ushort(source.sc),
                      _convert_bfloat16_as_ushort(source.sd),
                      _convert_bfloat16_as_ushort(source.se),
                      _convert_bfloat16_as_ushort(source.sf));
}

// --- as_bfloat16_float (ushort -> float) vectorized ---
inline float2 _convert_as_bfloat162_float2(ushort2 source) {
    return (float2)(_convert_as_bfloat16_float(source.s0),
                    _convert_as_bfloat16_float(source.s1));
}

inline float3 _convert_as_bfloat163_float3(ushort3 source) {
    return (float3)(_convert_as_bfloat16_float(source.s0),
                    _convert_as_bfloat16_float(source.s1),
                    _convert_as_bfloat16_float(source.s2));
}

inline float4 _convert_as_bfloat164_float4(ushort4 source) {
    return (float4)(_convert_as_bfloat16_float(source.s0),
                    _convert_as_bfloat16_float(source.s1),
                    _convert_as_bfloat16_float(source.s2),
                    _convert_as_bfloat16_float(source.s3));
}

inline float8 _convert_as_bfloat168_float8(ushort8 source) {
    return (float8)(_convert_as_bfloat16_float(source.s0),
                    _convert_as_bfloat16_float(source.s1),
                    _convert_as_bfloat16_float(source.s2),
                    _convert_as_bfloat16_float(source.s3),
                    _convert_as_bfloat16_float(source.s4),
                    _convert_as_bfloat16_float(source.s5),
                    _convert_as_bfloat16_float(source.s6),
                    _convert_as_bfloat16_float(source.s7));
}

inline float16 _convert_as_bfloat1616_float16(ushort16 source) {
    return (float16)(_convert_as_bfloat16_float(source.s0),
                     _convert_as_bfloat16_float(source.s1),
                     _convert_as_bfloat16_float(source.s2),
                     _convert_as_bfloat16_float(source.s3),
                     _convert_as_bfloat16_float(source.s4),
                     _convert_as_bfloat16_float(source.s5),
                     _convert_as_bfloat16_float(source.s6),
                     _convert_as_bfloat16_float(source.s7),
                     _convert_as_bfloat16_float(source.s8),
                     _convert_as_bfloat16_float(source.s9),
                     _convert_as_bfloat16_float(source.sa),
                     _convert_as_bfloat16_float(source.sb),
                     _convert_as_bfloat16_float(source.sc),
                     _convert_as_bfloat16_float(source.sd),
                     _convert_as_bfloat16_float(source.se),
                     _convert_as_bfloat16_float(source.sf));
}

#endif

// ===================== Identity overloads (shared by both paths) =====================
// Allow CONVERT_BFLOAT16_AS_USHORT to be called on values that are already ushort/ushortN.
inline ushort __attribute__((overloadable)) _convert_bfloat16_as_ushort(ushort val)
{
    return val;
}
inline ushort2 __attribute__((overloadable)) _convert_bfloat162_as_ushort2(ushort2 val)
{
    return val;
}
inline ushort3 __attribute__((overloadable)) _convert_bfloat163_as_ushort3(ushort3 val)
{
    return val;
}
inline ushort4 __attribute__((overloadable)) _convert_bfloat164_as_ushort4(ushort4 val)
{
    return val;
}
inline ushort8 __attribute__((overloadable)) _convert_bfloat168_as_ushort8(ushort8 val)
{
    return val;
}
inline ushort16 __attribute__((overloadable)) _convert_bfloat1616_as_ushort16(ushort16 val)
{
    return val;
}

// Scalar aliases so that size=1 concatenation resolves correctly.
#define _convert_bfloat161_as_ushort1(val) _convert_bfloat16_as_ushort(val)
#define _convert_as_bfloat161_float1(val)  _convert_as_bfloat16_float(val)

// Dispatch macros: concatenate size into the per-vector-width function names.
// Usage: CONVERT_BFLOAT16_AS_USHORT(val, 1)  -> _convert_bfloat16_as_ushort(val)
//        CONVERT_BFLOAT16_AS_USHORT(val, 4)  -> _convert_bfloat164_as_ushort4(val)
//        CONVERT_AS_BFLOAT16_FLOAT(val, 8)   -> _convert_as_bfloat168_float8(val)
#define CONVERT_BFLOAT16_AS_USHORT(val, size) CAT(_convert_bfloat16, CAT(size, CAT(_as_ushort, size)))(val)
#define CONVERT_AS_BFLOAT16_FLOAT(val, size)  CAT(_convert_as_bfloat16, CAT(size, CAT(_float, size)))(val)
