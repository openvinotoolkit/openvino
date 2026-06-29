// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "common.cl"

#ifdef intel_convert_as_bfloat16_float
#define _convert_as_bfloat16_float(val) intel_convert_as_bfloat16_float(val)
#else
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
#endif

#ifdef intel_convert_bfloat16_as_ushort
#define _convert_bfloat16_as_ushort(val) intel_convert_bfloat16_as_ushort(val)
#else
inline ushort _convert_bfloat16_as_ushort(float source) {
    uint* in = (uint*)&source;
    ushort u = 0;
    if ( (*in>>31) ) { 
        u = 1 << 15;
    }
    //exponent
    u += ( ( (*in >> 23) & 0b11111111)) << 7;
    //fraction
    u += (*in >> 16) & 0b1111111;
    return u;
}
#endif

// ===================== Vectorized bfloat16 <-> ushort helpers =====================

// --- bfloat16_as_ushort (float -> ushort) vectorized ---

#ifdef intel_convert_bfloat162_as_ushort2
#define _convert_bfloat162_as_ushort2(val) intel_convert_bfloat162_as_ushort2(val)
#else
inline ushort2 _convert_bfloat162_as_ushort2(float2 source) {
    return (ushort2)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1));
}
#endif

#ifdef intel_convert_bfloat163_as_ushort3
#define _convert_bfloat163_as_ushort3(val) intel_convert_bfloat163_as_ushort3(val)
#else
inline ushort3 _convert_bfloat163_as_ushort3(float3 source) {
    return (ushort3)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1),
                     _convert_bfloat16_as_ushort(source.s2));
}
#endif

#ifdef intel_convert_bfloat164_as_ushort4
#define _convert_bfloat164_as_ushort4(val) intel_convert_bfloat164_as_ushort4(val)
#else
inline ushort4 _convert_bfloat164_as_ushort4(float4 source) {
    return (ushort4)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1),
                     _convert_bfloat16_as_ushort(source.s2),
                     _convert_bfloat16_as_ushort(source.s3));
}
#endif

#ifdef intel_convert_bfloat168_as_ushort8
#define _convert_bfloat168_as_ushort8(val) intel_convert_bfloat168_as_ushort8(val)
#else
inline ushort8 _convert_bfloat168_as_ushort8(float8 source) {
    return (ushort8)(_convert_bfloat16_as_ushort(source.s0),
                     _convert_bfloat16_as_ushort(source.s1),
                     _convert_bfloat16_as_ushort(source.s2),
                     _convert_bfloat16_as_ushort(source.s3),
                     _convert_bfloat16_as_ushort(source.s4),
                     _convert_bfloat16_as_ushort(source.s5),
                     _convert_bfloat16_as_ushort(source.s6),
                     _convert_bfloat16_as_ushort(source.s7));
}
#endif

#ifdef intel_convert_bfloat1616_as_ushort16
#define _convert_bfloat1616_as_ushort16(val) intel_convert_bfloat1616_as_ushort16(val)
#else
inline ushort16 _convert_bfloat1616_as_ushort16(float16 source) {
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
#endif

// --- as_bfloat16_float (ushort -> float) vectorized ---

#ifdef intel_convert_as_bfloat162_float2
#define _convert_as_bfloat162_float2(val) intel_convert_as_bfloat162_float2(val)
#else
inline float2 _convert_as_bfloat162_float2(ushort2 source) {
    return (float2)(_convert_as_bfloat16_float(source.s0),
                    _convert_as_bfloat16_float(source.s1));
}
#endif

#ifdef intel_convert_as_bfloat163_float3
#define _convert_as_bfloat163_float3(val) intel_convert_as_bfloat163_float3(val)
#else
inline float3 _convert_as_bfloat163_float3(ushort3 source) {
    return (float3)(_convert_as_bfloat16_float(source.s0),
                    _convert_as_bfloat16_float(source.s1),
                    _convert_as_bfloat16_float(source.s2));
}
#endif

#ifdef intel_convert_as_bfloat164_float4
#define _convert_as_bfloat164_float4(val) intel_convert_as_bfloat164_float4(val)
#else
inline float4 _convert_as_bfloat164_float4(ushort4 source) {
    return (float4)(_convert_as_bfloat16_float(source.s0),
                    _convert_as_bfloat16_float(source.s1),
                    _convert_as_bfloat16_float(source.s2),
                    _convert_as_bfloat16_float(source.s3));
}
#endif

#ifdef intel_convert_as_bfloat168_float8
#define _convert_as_bfloat168_float8(val) intel_convert_as_bfloat168_float8(val)
#else
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
#endif

#ifdef intel_convert_as_bfloat1616_float16
#define _convert_as_bfloat1616_float16(val) intel_convert_as_bfloat1616_float16(val)
#else
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

// Scalar aliases so that size=1 concatenation resolves correctly.
#define _convert_bfloat161_as_ushort1(val) _convert_bfloat16_as_ushort(val)
#define _convert_as_bfloat161_float1(val)  _convert_as_bfloat16_float(val)

// Dispatch macros: concatenate size into the per-vector-width function names.
// Usage: CONVERT_BFLOAT16_AS_USHORT(val, 1)  -> _convert_bfloat16_as_ushort(val)
//        CONVERT_BFLOAT16_AS_USHORT(val, 4)  -> _convert_bfloat164_as_ushort4(val)
//        CONVERT_AS_BFLOAT16_FLOAT(val, 8)   -> _convert_as_bfloat168_float8(val)
#define CONVERT_BFLOAT16_AS_USHORT(val, size) CAT(_convert_bfloat16, CAT(size, CAT(_as_ushort, size)))(val)
#define CONVERT_AS_BFLOAT16_FLOAT(val, size)  CAT(_convert_as_bfloat16, CAT(size, CAT(_float, size)))(val)
