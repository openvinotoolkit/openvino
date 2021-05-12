// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512) || defined(HAVE_NEON)
  #define MANUAL_SIMD 1  // 1=call manually vectored code, 0=don't
#else
  #define MANUAL_SIMD 0
#endif

#ifdef HAVE_SSE
  #define USE_CVKL 1     // 1=reuse CVKL code for Resize, 0=don't
#else
  #define USE_CVKL 0
#endif

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512)
  #define CPU_SIMD 1
#else
  #define CPU_SIMD 0
#endif

#include <climits>
#include <cstdint>
#include <limits>

#if defined(__GNUC__) && (__GNUC__ <= 5)
#include <cmath>
#endif

namespace InferenceEngine {
namespace gapi {
namespace kernels {

template<typename DST, typename SRC> static inline DST saturate_cast(SRC x);
template<> inline short saturate_cast(int x) { return (std::min)(SHRT_MAX, (std::max)(SHRT_MIN, x)); }
template<> inline short saturate_cast(float x) { return saturate_cast<short>(static_cast<int>(std::rint(x))); }
template<> inline float saturate_cast(float x) { return x; }
template<> inline short saturate_cast(short x) { return x; }

template<> inline uint16_t saturate_cast(uint16_t x) { return x; }
template<> inline float    saturate_cast(uint16_t x) { return x; }

template<> inline uint16_t saturate_cast(int x) { return (std::min)(USHRT_MAX, (std::max)(0, x)); }
template<> inline uint16_t saturate_cast(float x)    { return saturate_cast<uint16_t>(static_cast<int>(std::rint(x))); }
template<> inline uchar saturate_cast<uchar>(int v)  { return (uchar)((unsigned)v <= UCHAR_MAX ? v : v > 0 ? UCHAR_MAX : 0); }

template<> inline uint16_t saturate_cast(uint8_t x) { return x; }
template<> inline float    saturate_cast(uint8_t x) { return x; }
template<> inline uint8_t  saturate_cast(uint8_t x) { return x; }

template<> inline uint8_t saturate_cast(uint16_t x) {
  using lim = std::numeric_limits<uint8_t>;
  return (uint8_t)std::min(static_cast<uint16_t>(lim::max()),
                           std::max(static_cast<uint16_t>(lim::min()), x));
}
template<> inline uint8_t saturate_cast(float x)    { return saturate_cast<uint8_t>(static_cast<int>(std::rint(x))); }

template<> inline float saturate_cast(double x)     { return x; }
//------------------------------------------------------------------------------

constexpr static const int ONE = 1 << 15;

inline static uint8_t calc(short alpha0, uint8_t src0, short alpha1, uint8_t src1) {
    constexpr static const int half = 1 << 14;
    return (src0 * alpha0 + src1 * alpha1 + half) >> 15;
}

inline static float calc(float alpha0, float src0, float alpha1, float src1) {
    return src0 * alpha0 + src1 * alpha1;
}

//------------------------------------------------------------------------------

// Variants:
// - F=float, I=int
// - F=short, I=short (e.g. F is Q1.7.8 encoded with short)
template<typename F, typename I>
struct MapperUnit {
    F alpha0, alpha1;
    I index0, index1;
};

//------------------------------------------------------------------------------

typedef uint16_t Q0_16;  // value [0..1)   with 16 fractional bits
typedef uint16_t Q8_8;   // value [0..255) with  8 fractional bits
typedef uint8_t  U8;     // value [0..255)

typedef MapperUnit<float,   int> MapperUnit32F;
typedef MapperUnit<Q0_16, short> MapperUnit8U;

template<typename DST, typename SRC> static inline DST convert_cast(SRC x);
template<> inline uint8_t convert_cast(uint8_t x) { return x; }
template<> inline uint8_t convert_cast(float x) { return static_cast<uint8_t>(x); }
template<> inline float convert_cast(float  x) { return x; }
template<> inline float convert_cast(double x) { return static_cast<float>(x); }
template<> inline Q0_16 convert_cast(double x) {
    int ix = static_cast<int>(std::rint(x * (1 << 16)));
    return saturate_cast<Q0_16>(ix);
}
template<> inline Q8_8 convert_cast(uchar x) { return x << 8; }
template<> inline uchar convert_cast(Q8_8 x) { return x >> 8; }

template<typename DST, typename SRC> static inline DST checked_cast(SRC x) {
    short dx = static_cast<DST>(x);
    GAPI_Assert(x == dx);  // check
    return dx;
}

static inline Q8_8 mulas(Q0_16 a, U8   s) { return static_cast<Q8_8>((a * s) >>  8); }
static inline Q8_8 mulaw(Q0_16 a, Q8_8 w) { return static_cast<Q8_8>((a * w) >> 16); }

static inline float mulas(float a, float s) { return a * s; }
static inline float mulaw(float a, float w) { return a * w; }

}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine
