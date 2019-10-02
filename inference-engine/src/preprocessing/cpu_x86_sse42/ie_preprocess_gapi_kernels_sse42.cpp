// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>

#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_gapi_kernels_impl.hpp"
#include "ie_preprocess_gapi_kernels_sse42.hpp"

// NB: include this before opencv_hal_sse.hpp
#include "nmmintrin.h"

// NB: define these before opencv_hal_sse.hpp
namespace cv {
namespace hal {

enum StoreMode {
    STORE_UNALIGNED = 0,
    STORE_ALIGNED = 1,
    STORE_ALIGNED_NOCACHE = 2
};

}  // namespace hal
}  // namespace cv

// NB: define these before opencv_hal_sse.hpp
#define OPENCV_HAL_ADD(a, b) ((a) + (b))
#define OPENCV_HAL_AND(a, b) ((a) & (b))
#define OPENCV_HAL_NOP(a) (a)
#define OPENCV_HAL_1ST(a, b) (a)

// NB: define these before opencv_hal_sse.hpp
#ifdef CV_SSE4_2
  #undef CV_SSE4_2
  #undef CV_SSE4_1
  #undef CV_SSSE3
  #undef CV_SSE3
  #undef CV_SSE2
  #undef CV_SSE
#endif
#define CV_SSE4_2 1
#define CV_SSE4_1 1
#define CV_SSSE3  1
#define CV_SSE3   1
#define CV_SSE2   1
#define CV_SSE    1
#define CV_CPU_HAS_SUPPORT_SSE2 1
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN  // empty
#define CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

// OpenCV universal intrinsic
#include "opencv_hal_sse.hpp"

// AFTER "opencv_hal_sse.hpp"
// (CV_SIMD128 defined there)
#if   !CV_SIMD128
#error CV_SIMD128 is required!
#endif

#include <cstring>

using namespace cv;

namespace InferenceEngine {
namespace gapi {
namespace kernels {

//----------------------------------------------------------------------

#if CV_SSE
static inline void v_deinterleave(const v_float32x4& low, const v_float32x4& high,
                                        v_float32x4& even,      v_float32x4& odd) {
    __m128 tmp0 = _mm_unpacklo_ps(low.val, high.val);
    __m128 tmp1 = _mm_unpackhi_ps(low.val, high.val);
    even.val = _mm_unpacklo_ps(tmp0, tmp1);
    odd .val = _mm_unpackhi_ps(tmp0, tmp1);
}
#endif

#if CV_SSE2
static inline void v_deinterleave(const v_uint8x16& i0, const v_uint8x16& i1,
                                  const v_uint8x16& i2, const v_uint8x16& i3,
                                        v_uint8x16& o0,       v_uint8x16& o1,
                                        v_uint8x16& o2,       v_uint8x16& o3) {
    __m128i u0 = i0.val;                     // a0 b0 c0 d0 a1 b1 c1 d1 ...
    __m128i u1 = i1.val;                     // a4 b4 c4 d4 ...
    __m128i u2 = i2.val;                     // a8 b8 c8 d8 ...
    __m128i u3 = i3.val;                     // a12 b12 c12 d12 ...

    __m128i v0 = _mm_unpacklo_epi8(u0, u2);  // a0 a8 b0 b8 ...
    __m128i v1 = _mm_unpackhi_epi8(u0, u2);  // a2 a10 b2 b10 ...
    __m128i v2 = _mm_unpacklo_epi8(u1, u3);  // a4 a12 b4 b12 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3);  // a6 a14 b6 b14 ...

    u0 = _mm_unpacklo_epi8(v0, v2);          // a0 a4 a8 a12 ...
    u1 = _mm_unpacklo_epi8(v1, v3);          // a2 a6 a10 a14 ...
    u2 = _mm_unpackhi_epi8(v0, v2);          // a1 a5 a9 a13 ...
    u3 = _mm_unpackhi_epi8(v1, v3);          // a3 a7 a11 a15 ...

    v0 = _mm_unpacklo_epi8(u0, u1);          // a0 a2 a4 a6 ...
    v1 = _mm_unpacklo_epi8(u2, u3);          // a1 a3 a5 a7 ...
    v2 = _mm_unpackhi_epi8(u0, u1);          // c0 c2 c4 c6 ...
    v3 = _mm_unpackhi_epi8(u2, u3);          // c1 c3 c5 c7 ...

    o0.val = _mm_unpacklo_epi8(v0, v1);      // a0 a1 a2 a3 ...
    o1.val = _mm_unpackhi_epi8(v0, v1);      // b0 b1 b2 b3 ...
    o2.val = _mm_unpacklo_epi8(v2, v3);      // c0 c1 c2 c3 ...
    o3.val = _mm_unpackhi_epi8(v2, v3);      // d0 d1 d2 d3 ...
}

static inline v_uint8x16 v_interleave_low(const v_uint8x16& a, const v_uint8x16& b) {
    return v_uint8x16(_mm_unpacklo_epi8(a.val, b.val));
}

static inline v_uint8x16 v_interleave_high(const v_uint8x16& a, const v_uint8x16& b) {
    return v_uint8x16(_mm_unpackhi_epi8(a.val, b.val));
}

static inline v_int16x8 v_interleave_low(const v_int16x8& a, const v_int16x8& b) {
    return v_int16x8(_mm_unpacklo_epi16(a.val, b.val));
}

static inline v_int16x8 v_interleave_high(const v_int16x8& a, const v_int16x8& b) {
    return v_int16x8(_mm_unpackhi_epi16(a.val, b.val));
}

static inline v_uint16x8 v_expand_low(const v_uint8x16& a) {
    return v_uint16x8(_mm_unpacklo_epi8(a.val, _mm_setzero_si128()));
}

static inline v_uint16x8 v_expand_high(const v_uint8x16& a) {
    return v_uint16x8(_mm_unpackhi_epi8(a.val, _mm_setzero_si128()));
}

static inline v_uint8x16 v_saturate_u8(const v_int16x8& a) {
    v_uint8x16 r;
    r.val = _mm_packus_epi16(a.val, _mm_setzero_si128());
    return r;
}

static inline v_int16x8 v_saturate_s16(const v_int32x4& a) {
    v_int16x8 r;
    r.val = _mm_packs_epi32(a.val, _mm_setzero_si128());
    return r;
}

// for each j=index[k], load two chars src[j] and src[j+1]
static inline v_uint8x16 v_gather_pairs(const uchar src[], const v_int16x8& index) {
    v_uint8x16 r;
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 0)]), 0);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 1)]), 1);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 2)]), 2);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 3)]), 3);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 4)]), 4);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 5)]), 5);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 6)]), 6);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const ushort*>(&src[_mm_extract_epi16(index.val, 7)]), 7);
    return r;
}

static inline v_int16x8 v_gather_chan(const uchar src[], const v_int16x8& index, int channel, int pos) {
    constexpr const int chanNum = 3;
    v_int16x8 r;
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 0) + pos) + channel]), 0);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 1) + pos) + channel]), 1);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 2) + pos) + channel]), 2);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 3) + pos) + channel]), 3);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 4) + pos) + channel]), 4);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 5) + pos) + channel]), 5);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 6) + pos) + channel]), 6);
    r.val = _mm_insert_epi16(r.val, *reinterpret_cast<const uchar*>(&src[chanNum*(_mm_extract_epi16(index.val, 7) + pos) + channel]), 7);
    return r;
}

static inline void v_gather_pairs(const float src[], const v_int32x4& index,
                                  v_float32x4& low, v_float32x4& high) {
    int i[4];
    v_store(i, index);

    __m128 l = _mm_setzero_ps();
    l = _mm_loadl_pi(l, (const __m64*)&src[i[0]]);  // pair of floats
    l = _mm_loadh_pi(l, (const __m64*)&src[i[1]]);
    low.val = l;

    __m128 h = _mm_setzero_ps();
    h = _mm_loadl_pi(h, (const __m64*)&src[i[2]]);
    h = _mm_loadh_pi(h, (const __m64*)&src[i[3]]);
    high.val = h;
}

static inline v_int32x4 v_madd(const v_int16x8& a, const v_int16x8& b) {
    v_int32x4 r;
    r.val = _mm_madd_epi16(a.val, b.val);
    return r;
}

static inline v_int16x8 v_mulhi(const v_int16x8& a, short b) {
    v_int16x8 r;
    r.val = _mm_mulhi_epi16(a.val, _mm_set1_epi16(b));
    return r;
}

static inline v_uint16x8 v_mulhi(const v_uint16x8& a, v_uint16x8 b) {
    v_uint16x8 r;
    r.val = _mm_mulhi_epu16(a.val, b.val);
    return r;
}

static inline v_uint16x8 v_mulhi(const v_uint16x8& a, uint16_t b) {
    v_uint16x8 r;
    r.val = _mm_mulhi_epu16(a.val, _mm_set1_epi16(b));
    return r;
}

static inline v_int16x8 v_mulhrs(const v_int16x8& a, const v_int16x8& b) {
    v_int16x8 r;
    r.val = _mm_mulhrs_epi16(a.val, b.val);
    return r;
}

static inline v_int16x8 v_mulhrs(const v_int16x8& a, short b) {
    return v_mulhrs(a, v_setall_s16(b));
}
#endif  // SSE2

#ifdef CV_SSE3
static inline void v_deinterleave_expand(const v_uint8x16& src, v_int16x8& even, v_int16x8& odd) {
    static const __m128i mask_even = _mm_setr_epi8(0, -1, 2, -1, 4, -1, 6, -1, 8, -1, 10, -1, 12, -1, 14, -1);
    static const __m128i mask_odd  = _mm_setr_epi8(1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 11, -1, 13, -1, 15, -1);
    even.val = _mm_shuffle_epi8(src.val, mask_even);
    odd .val = _mm_shuffle_epi8(src.val, mask_odd);
}
#endif

static inline v_float32x4 v_fma(const v_float32x4& a, float b, const v_float32x4& c) {
    return v_fma(a, v_setall_f32(b), c);
}

static inline v_int16x8 operator+ (const v_int16x8& a, short b) {
    return a + v_setall_s16(b);
}

static inline v_int16x8 operator- (short a, const v_int16x8& b) {
    return v_setall_s16(a) - b;
}

static inline v_float32x4 operator- (float a, const v_float32x4& b) {
    return v_setall_f32(a) - b;
}

static inline v_float32x4 operator* (const v_float32x4& a, float b) {
    return a * v_setall_f32(b);
}

//------------------------------------------------------------------------------

// Resize (bi-linear, 8U)
void calcRowLinear_8U(uint8_t *dst[],
                const uint8_t *src0[],
                const uint8_t *src1[],
                const short    alpha[],
                const short    clone[],  // 4 clones of alpha
                const short    mapsx[],
                const short    beta[],
                      uint8_t  tmp[],
                const Size   & inSz,
                const Size   & outSz,
                      int      lpi) {
    bool xRatioEq1 = inSz.width  == outSz.width;
    bool yRatioEq1 = inSz.height == outSz.height;

    if (!xRatioEq1 && !yRatioEq1) {
        if (4 == lpi) {
            // vertical pass
            GAPI_DbgAssert(inSz.width >= 8);

            __m128i b0 = _mm_set1_epi16(beta[0]);
            __m128i b1 = _mm_set1_epi16(beta[1]);
            __m128i b2 = _mm_set1_epi16(beta[2]);
            __m128i b3 = _mm_set1_epi16(beta[3]);

            for (int w = 0; w < inSz.width; ) {
                for (; w <= inSz.width - 8; w += 8) {
                #if USE_CVKL
                    //--------------------------------------------
                    // reworked from: ie_preprocess_data_sse42.cpp
                    //      function: resize_bilinear_u8
                    //         label: vertical_pass
                    //--------------------------------------------

                    __m128i val0lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src0[0][w])),
                                                                     *reinterpret_cast<const int64_t*>(&src0[1][w]), 1);
                    __m128i val0hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src0[2][w])),
                                                                     *reinterpret_cast<const int64_t*>(&src0[3][w]), 1);
                    __m128i val1lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src1[0][w])),
                                                                     *reinterpret_cast<const int64_t*>(&src1[1][w]), 1);
                    __m128i val1hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src1[2][w])),
                                                                     *reinterpret_cast<const int64_t*>(&src1[3][w]), 1);

                    __m128i val0_0 = _mm_cvtepu8_epi16(val0lo);
                    __m128i val0_2 = _mm_cvtepu8_epi16(val0hi);
                    __m128i val1_0 = _mm_cvtepu8_epi16(val1lo);
                    __m128i val1_2 = _mm_cvtepu8_epi16(val1hi);

                    __m128i val0_1 = _mm_unpackhi_epi8(val0lo, _mm_setzero_si128());
                    __m128i val0_3 = _mm_unpackhi_epi8(val0hi, _mm_setzero_si128());
                    __m128i val1_1 = _mm_unpackhi_epi8(val1lo, _mm_setzero_si128());
                    __m128i val1_3 = _mm_unpackhi_epi8(val1hi, _mm_setzero_si128());

                    __m128i t0 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_0, val1_0), b0);
                    __m128i t1 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_1, val1_1), b1);
                    __m128i t2 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_2, val1_2), b2);
                    __m128i t3 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_3, val1_3), b3);

                    __m128i r0 = _mm_add_epi16(val1_0, t0);
                    __m128i r1 = _mm_add_epi16(val1_1, t1);
                    __m128i r2 = _mm_add_epi16(val1_2, t2);
                    __m128i r3 = _mm_add_epi16(val1_3, t3);

                    __m128i q0 = _mm_packus_epi16(r0, r1);
                    __m128i q1 = _mm_packus_epi16(r2, r3);

                    __m128i q2 = _mm_blend_epi16(q0, _mm_slli_si128(q1, 4), 0xCC /*0b11001100*/);
                    __m128i q3 = _mm_blend_epi16(_mm_srli_si128(q0, 4), q1, 0xCC /*0b11001100*/);

                    __m128i q4 = _mm_shuffle_epi8(q2, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));
                    __m128i q5 = _mm_shuffle_epi8(q3, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));

                    _mm_storeu_si128(reinterpret_cast<__m128i *>(&tmp[4*w +  0]), q4);
                    _mm_storeu_si128(reinterpret_cast<__m128i *>(&tmp[4*w + 16]), q5);

                #else
                    // let: t[i] = src0[i][w]*beta0[i] + src1[i][w]*beta1
                    // here: beta0[i] = beta[i], beta1 = 1 - beta0[i]
                    v_int16x8 t0, t1, t2, t3;
                    {
                        v_int16x8 s0, s1;

                        s0 = v_reinterpret_as_s16(v_load_expand(&src0[0][w]));
                        s1 = v_reinterpret_as_s16(v_load_expand(&src1[0][w]));
                        t0 = v_mulhrs(s0 - s1, beta[0]) + s1;

                        s0 = v_reinterpret_as_s16(v_load_expand(&src0[1][w]));
                        s1 = v_reinterpret_as_s16(v_load_expand(&src1[1][w]));
                        t1 = v_mulhrs(s0 - s1, beta[1]) + s1;

                        s0 = v_reinterpret_as_s16(v_load_expand(&src0[2][w]));
                        s1 = v_reinterpret_as_s16(v_load_expand(&src1[2][w]));
                        t2 = v_mulhrs(s0 - s1, beta[2]) + s1;

                        s0 = v_reinterpret_as_s16(v_load_expand(&src0[3][w]));
                        s1 = v_reinterpret_as_s16(v_load_expand(&src1[3][w]));
                        t3 = v_mulhrs(s0 - s1, beta[3]) + s1;
                    }
                    // store as groups of 4 pixels: each group to have a pixel per row
                    {
                        v_uint8x16 a0, a1, a2, a3;
                        a0 = v_pack_u(t0, v_setall_s16(0));
                        a1 = v_pack_u(t1, v_setall_s16(0));
                        a2 = v_pack_u(t2, v_setall_s16(0));
                        a3 = v_pack_u(t3, v_setall_s16(0));

                        v_int16x8 b0, b1;
                        b0 = v_reinterpret_as_s16(v_interleave_low(a0, a1));  // 0th, 1st
                        b1 = v_reinterpret_as_s16(v_interleave_low(a2, a3));  // 2nd, 3rd

                        v_uint8x16 d0, d1;
                        d0 = v_reinterpret_as_u8(v_interleave_low(b0,  b1));
                        d1 = v_reinterpret_as_u8(v_interleave_high(b0, b1));

                        v_store(&tmp[4*w +  0], d0);
                        v_store(&tmp[4*w + 16], d1);
                    }
                #endif
                }

                if (w < inSz.width) {
                    w = inSz.width - 8;
                }
            }

            // horizontal pass
            GAPI_DbgAssert(outSz.width >= 8);
            for (int x = 0; x < outSz.width; ) {
                for (; x <= outSz.width - 8; x += 8) {
                #if USE_CVKL
                    //--------------------------------------------
                    // reworked from: ie_preprocess_data_sse42.cpp
                    //      function: resize_bilinear_u8
                    //         label: horizontal_pass
                    //--------------------------------------------

                #if 1
                    __m128i a10 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 *  x]));
                    __m128i a32 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * (x + 2)]));
                    __m128i a54 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * (x + 4)]));
                    __m128i a76 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * (x + 6)]));
                #else
                    // provided alpha[x..x+7] = { a0, a1, a2, a3, a4, a5, a6, a7},
                    // clone each a[i] 4 times - one item per each of LPI rows,
                    // so that a10 = {a0, a0, a0, a0, a1, a1, a1, a1}, etc.
                    __m128i a10, a32, a54, a76;
                    __m128i alpha0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&alpha[x]));
                    a10 = _mm_unpacklo_epi16(alpha0, alpha0);  // {a0, a0, a1, a1, a2, a2, a3, a3}
                    a32 = _mm_unpackhi_epi16(a10, a10);        // {a2, a2, a2, a2, a3, a3, a3, a3}
                    a10 = _mm_unpacklo_epi16(a10, a10);        // {a0, a0, a0, a0, a1, a1, a1, a1}
                    a54 = _mm_unpackhi_epi16(alpha0, alpha0);  // {a4, a4, a5, a5, a6, a6, a7, a7}
                    a76 = _mm_unpackhi_epi16(a54, a54);        // {a6, a6, a6, a6, a7, a7, a7, a7}
                    a54 = _mm_unpacklo_epi16(a54, a54);        // {a4, a4, a4, a4, a5, a5, a5, a5}
                #endif

                    __m128d val0d, val1d, val2d, val3d;
                    val0d = _mm_load_sd(/****/  reinterpret_cast<double*>(&tmp[4 * mapsx[x + 0]]));
                    val0d = _mm_loadh_pd(val0d, reinterpret_cast<double*>(&tmp[4 * mapsx[x + 1]]));
                    val1d = _mm_load_sd(/****/  reinterpret_cast<double*>(&tmp[4 * mapsx[x + 2]]));
                    val1d = _mm_loadh_pd(val1d, reinterpret_cast<double*>(&tmp[4 * mapsx[x + 3]]));
                    val2d = _mm_load_sd(/****/  reinterpret_cast<double*>(&tmp[4 * mapsx[x + 4]]));
                    val2d = _mm_loadh_pd(val2d, reinterpret_cast<double*>(&tmp[4 * mapsx[x + 5]]));
                    val3d = _mm_load_sd(/****/  reinterpret_cast<double*>(&tmp[4 * mapsx[x + 6]]));
                    val3d = _mm_loadh_pd(val3d, reinterpret_cast<double*>(&tmp[4 * mapsx[x + 7]]));

                    __m128i val_0 = _mm_castpd_si128(val0d);
                    __m128i val_1 = _mm_castpd_si128(val1d);
                    __m128i val_2 = _mm_castpd_si128(val2d);
                    __m128i val_3 = _mm_castpd_si128(val3d);

                    val_0 = _mm_shuffle_epi32(val_0, _MM_SHUFFLE(3, 1, 2, 0));
                    val_1 = _mm_shuffle_epi32(val_1, _MM_SHUFFLE(3, 1, 2, 0));
                    val_2 = _mm_shuffle_epi32(val_2, _MM_SHUFFLE(3, 1, 2, 0));
                    val_3 = _mm_shuffle_epi32(val_3, _MM_SHUFFLE(3, 1, 2, 0));

                    __m128i val0_0 = _mm_cvtepu8_epi16(val_0);
                    __m128i val0_1 = _mm_cvtepu8_epi16(val_1);
                    __m128i val0_2 = _mm_cvtepu8_epi16(val_2);
                    __m128i val0_3 = _mm_cvtepu8_epi16(val_3);

                    __m128i val1_0 = _mm_unpackhi_epi8(val_0, _mm_setzero_si128());
                    __m128i val1_1 = _mm_unpackhi_epi8(val_1, _mm_setzero_si128());
                    __m128i val1_2 = _mm_unpackhi_epi8(val_2, _mm_setzero_si128());
                    __m128i val1_3 = _mm_unpackhi_epi8(val_3, _mm_setzero_si128());

                    __m128i t0 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_0, val1_0), a10);
                    __m128i t1 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_1, val1_1), a32);
                    __m128i t2 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_2, val1_2), a54);
                    __m128i t3 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_3, val1_3), a76);

                    __m128i r0 = _mm_add_epi16(val1_0, t0);
                    __m128i r1 = _mm_add_epi16(val1_1, t1);
                    __m128i r2 = _mm_add_epi16(val1_2, t2);
                    __m128i r3 = _mm_add_epi16(val1_3, t3);

                    __m128i q0 = _mm_packus_epi16(r0, r1);
                    __m128i q1 = _mm_packus_epi16(r2, r3);

                    __m128i q2 = _mm_shuffle_epi8(q0, _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15));
                    __m128i q3 = _mm_shuffle_epi8(q1, _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15));

                    __m128i q4 = _mm_blend_epi16(q2, _mm_slli_si128(q3, 4), 0xCC /*0b11001100*/);
                    __m128i q5 = _mm_blend_epi16(_mm_srli_si128(q2, 4), q3, 0xCC /*0b11001100*/);

                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[0][x]),                q4);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[1][x]), _mm_srli_si128(q4, 8));
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[2][x]),                q5);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[3][x]), _mm_srli_si128(q5, 8));

                #else
                    // let: t be 2 pairs of groups of 4 pixels (each group is for 4 dst rows)
                    // each pair of gorups corresponds to pixels indexed as sx0 and sx1=sx0+1
                    // so: low part of t0 is 2x4 pixels corresponding to sx0=mapsx[x+0], etc.
                    v_uint8x16 t0, t1, t2, t3;
                    {
                        t0.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 0]])),
                                                                 *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 1]]), 1);
                        t1.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 2]])),
                                                                 *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 3]]), 1);
                        t2.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 4]])),
                                                                 *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 5]]), 1);
                        t3.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 6]])),
                                                                 *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 7]]), 1);
                    }

                    // let: r0 be pixels for 0th row, etc
                    v_uint8x16 r0, r1, r2, r3;
                    v_deinterleave(t0, t1, t2, t3, r0, r1, r2, r3);

                    // let: dl be resulting 8 pixels for l'th row
                    //      dl = alpha0*s0l + alpha1*s1l
                    // note that alpha0 + alpha1 = 1
                    {
                        v_int16x8 s0, s1, d, alpha0;

                        alpha0 = v_load(&alpha[x]);  // 8 coefficients

                        v_deinterleave_expand(r0, s0, s1);
                        d = v_mulhrs(s0 - s1, alpha0) + s1;
                        v_pack_u_store(&dst[0][x], d);

                        v_deinterleave_expand(r1, s0, s1);
                        d = v_mulhrs(s0 - s1, alpha0) + s1;
                        v_pack_u_store(&dst[1][x], d);

                        v_deinterleave_expand(r2, s0, s1);
                        d = v_mulhrs(s0 - s1, alpha0) + s1;
                        v_pack_u_store(&dst[2][x], d);

                        v_deinterleave_expand(r3, s0, s1);
                        d = v_mulhrs(s0 - s1, alpha0) + s1;
                        v_pack_u_store(&dst[3][x], d);
                    }
                #endif
                }

                if (x < outSz.width) {
                    x = outSz.width - 8;
                }
            }

        } else {  // if any lpi
            for (int l = 0; l < lpi; l++) {
                short beta0 =                            beta[l];
            //  short beta1 = saturate_cast<short>(ONE - beta[l]);

                // vertical pass
                GAPI_DbgAssert(inSz.width >= 8);
                for (int w = 0; w < inSz.width; ) {
                    for (; w <= inSz.width - 8; w += 8) {
                        v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(&src0[l][w]));
                        v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(&src1[l][w]));
                        v_int16x8 t = v_mulhrs(s0 - s1, beta0) + s1;
                        v_pack_u_store(tmp + w, t);
                    }

                    if (w < inSz.width) {
                        w = inSz.width - 8;
                    }
                }

                // horizontal pass
                GAPI_DbgAssert(outSz.width >= 8);
                for (int x = 0; x < outSz.width; ) {
                    for (; x <= outSz.width - 8; x += 8) {
                        v_int16x8 a0 = v_load(&alpha[x]);        // as signed Q1.1.14
                        v_int16x8 sx = v_load(&mapsx[x]);        // as integer (int16)
                        v_uint8x16 t = v_gather_pairs(tmp, sx);  // 8 pairs of src0 pixels
                        v_int16x8 t0, t1;
                        v_deinterleave_expand(t, t0, t1);        // tmp pixels as int16
                        v_int16x8 d = v_mulhrs(t0 - t1, a0) + t1;
                        v_pack_u_store(&dst[l][x], d);
                    }

                    if (x < outSz.width) {
                        x = outSz.width - 8;
                    }
                }
            }
        }  // if lpi == 4

    } else if (!xRatioEq1) {
        GAPI_DbgAssert(yRatioEq1);

        if (4 == lpi) {
            // vertical pass
            GAPI_DbgAssert(inSz.width >= 16);
            for (int w = 0; w < inSz.width; ) {
                for (; w <= inSz.width - 16; w += 16) {
                    v_uint8x16 s0, s1, s2, s3;
                    s0 = v_load(&src0[0][w]);
                    s1 = v_load(&src0[1][w]);
                    s2 = v_load(&src0[2][w]);
                    s3 = v_load(&src0[3][w]);
                    v_store_interleave(&tmp[4*w], s0, s1, s2, s3);
                }

                if (w < inSz.width) {
                    w = inSz.width - 16;
                }
            }

            // horizontal pass
            GAPI_DbgAssert(outSz.width >= 8);
            for (int x = 0; x < outSz.width; ) {
                for (; x <= outSz.width - 8; x += 8) {
                    v_uint8x16 t0, t1, t2, t3;
                    t0.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 0]])),
                                                             *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 1]]), 1);
                    t1.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 2]])),
                                                             *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 3]]), 1);
                    t2.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 4]])),
                                                             *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 5]]), 1);
                    t3.val = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<__m128i*>(&tmp[4 * mapsx[x + 6]])),
                                                             *reinterpret_cast<int64_t*>(&tmp[4 * mapsx[x + 7]]), 1);

                    v_uint8x16 r0, r1, r2, r3;
                    v_deinterleave(t0, t1, t2, t3, r0, r1, r2, r3);

                    v_int16x8 s0, s1, d, alpha0;

                    alpha0 = v_load(&alpha[x]);  // 8 coefficients

                    v_deinterleave_expand(r0, s0, s1);
                    d = v_mulhrs(s0 - s1, alpha0) + s1;
                    v_pack_u_store(&dst[0][x], d);

                    v_deinterleave_expand(r1, s0, s1);
                    d = v_mulhrs(s0 - s1, alpha0) + s1;
                    v_pack_u_store(&dst[1][x], d);

                    v_deinterleave_expand(r2, s0, s1);
                    d = v_mulhrs(s0 - s1, alpha0) + s1;
                    v_pack_u_store(&dst[2][x], d);

                    v_deinterleave_expand(r3, s0, s1);
                    d = v_mulhrs(s0 - s1, alpha0) + s1;
                    v_pack_u_store(&dst[3][x], d);
                }

                if (x < outSz.width) {
                    x = outSz.width - 8;
                }
            }

        } else {  // any LPI
            for (int l = 0; l < lpi; l++) {
                const uchar *src = src0[l];

                // horizontal pass
                GAPI_DbgAssert(outSz.width >= 8);
                for (int x = 0; x < outSz.width; ) {
                    for (; x <= outSz.width - 8; x += 8) {
                        v_int16x8 a0 = v_load(&alpha[x]);        // as signed Q1.1.14
                        v_int16x8 sx = v_load(&mapsx[x]);        // as integer (int16)
                        v_uint8x16 t = v_gather_pairs(src, sx);  // 8 pairs of src0 pixels
                        v_int16x8 t0, t1;
                        v_deinterleave_expand(t, t0, t1);        // tmp pixels as int16
                        v_int16x8 d = v_mulhrs(t0 - t1, a0) + t1;
                        v_pack_u_store(&dst[l][x], d);
                    }

                    if (x < outSz.width) {
                        x = outSz.width - 8;
                    }
                }
            }
        }

    } else if (!yRatioEq1) {
        GAPI_DbgAssert(xRatioEq1);
        int length = inSz.width;  // == outSz.width

        for (int l = 0; l < lpi; l++) {
            short beta0 =                            beta[l];
        //  short beta1 = saturate_cast<short>(ONE - beta[l]);

            // vertical pass
            GAPI_DbgAssert(inSz.width >= 8);
            for (int w = 0; w < outSz.width; ) {
                for (; w <= length - 8; w += 8) {
                    v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(src0[l] + w));
                    v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(src1[l] + w));
                    v_int16x8 t = v_mulhrs(s0 - s1, beta0) + s1;
                    v_pack_u_store(dst[l] + w, t);
                }

                if (w < inSz.width) {
                    w = inSz.width - 8;
                }
            }
        }

    } else {
        GAPI_DbgAssert(xRatioEq1 && yRatioEq1);
        int length = inSz.width;  // == outSz.width

        for (int l = 0; l < lpi; l++) {
            memcpy(dst[l], src0[l], length);
        }
    }
}

// Resize (bi-linear, 8UC3)
void calcRowLinear_8UC3(std::array<std::array<uint8_t*, 4>, 3> &dst,
                  const uint8_t *src0[],
                  const uint8_t *src1[],
                  const short    alpha[],
                  const short    clone[],  // 4 clones of alpha
                  const short    mapsx[],
                  const short    beta[],
                        uint8_t  tmp[],
                  const Size    &inSz,
                  const Size    &outSz,
                        int      lpi) {
    constexpr const int chanNum = 3;

    if (4 == lpi) {
        // vertical pass
        GAPI_DbgAssert(inSz.width >= 8);

        __m128i b0 = _mm_set1_epi16(beta[0]);
        __m128i b1 = _mm_set1_epi16(beta[1]);
        __m128i b2 = _mm_set1_epi16(beta[2]);
        __m128i b3 = _mm_set1_epi16(beta[3]);

        for (int w = 0; w < inSz.width*chanNum; ) {
            for (; w <= inSz.width*chanNum - 8; w += 8) {
                //--------------------------------------------
                // reworked from: ie_preprocess_data_sse42.cpp
                //      function: resize_bilinear_u8
                //         label: vertical_pass
                //--------------------------------------------

                __m128i val0lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src0[0][w])),
                        *reinterpret_cast<const int64_t*>(&src0[1][w]), 1);
                __m128i val0hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src0[2][w])),
                        *reinterpret_cast<const int64_t*>(&src0[3][w]), 1);
                __m128i val1lo = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src1[0][w])),
                        *reinterpret_cast<const int64_t*>(&src1[1][w]), 1);
                __m128i val1hi = _mm_insert_epi64(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(&src1[2][w])),
                        *reinterpret_cast<const int64_t*>(&src1[3][w]), 1);

                __m128i val0_0 = _mm_cvtepu8_epi16(val0lo);
                __m128i val0_2 = _mm_cvtepu8_epi16(val0hi);
                __m128i val1_0 = _mm_cvtepu8_epi16(val1lo);
                __m128i val1_2 = _mm_cvtepu8_epi16(val1hi);

                __m128i val0_1 = _mm_unpackhi_epi8(val0lo, _mm_setzero_si128());
                __m128i val0_3 = _mm_unpackhi_epi8(val0hi, _mm_setzero_si128());
                __m128i val1_1 = _mm_unpackhi_epi8(val1lo, _mm_setzero_si128());
                __m128i val1_3 = _mm_unpackhi_epi8(val1hi, _mm_setzero_si128());

                __m128i t0 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_0, val1_0), b0);
                __m128i t1 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_1, val1_1), b1);
                __m128i t2 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_2, val1_2), b2);
                __m128i t3 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_3, val1_3), b3);

                __m128i r0 = _mm_add_epi16(val1_0, t0);
                __m128i r1 = _mm_add_epi16(val1_1, t1);
                __m128i r2 = _mm_add_epi16(val1_2, t2);
                __m128i r3 = _mm_add_epi16(val1_3, t3);

                __m128i q0 = _mm_packus_epi16(r0, r1);
                __m128i q1 = _mm_packus_epi16(r2, r3);

                __m128i q2 = _mm_blend_epi16(q0, _mm_slli_si128(q1, 4), 0xCC /*0b11001100*/);
                __m128i q3 = _mm_blend_epi16(_mm_srli_si128(q0, 4), q1, 0xCC /*0b11001100*/);

                __m128i q4 = _mm_shuffle_epi8(q2, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));
                __m128i q5 = _mm_shuffle_epi8(q3, _mm_setr_epi8(0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15));

                _mm_storeu_si128(reinterpret_cast<__m128i *>(&tmp[4*w +  0]), q4);
                _mm_storeu_si128(reinterpret_cast<__m128i *>(&tmp[4*w + 16]), q5);
            }

            if (w < inSz.width*chanNum) {
                w = inSz.width*chanNum - 8;
            }
        }

        // horizontal pass
        GAPI_DbgAssert(outSz.width >= 8);
        for (int x = 0; x < outSz.width; ) {
            for (; x <= outSz.width - 8; x += 8) {
                //--------------------------------------------
                // reworked from: ie_preprocess_data_sse42.cpp
                //      function: resize_bilinear_u8
                //         label: horizontal_pass
                //--------------------------------------------

                __m128i a10 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 *  x]));
                __m128i a32 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * (x + 2)]));
                __m128i a54 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * (x + 4)]));
                __m128i a76 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&clone[4 * (x + 6)]));

                __m128i val_0 = _mm_setzero_si128();
                __m128i val_1 = _mm_setzero_si128();
                __m128i val_2 = _mm_setzero_si128();
                __m128i val_3 = _mm_setzero_si128();

                for (int c = 0; c < chanNum; c++) {
                    val_0 = _mm_insert_epi32(val_0, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 0]      + c)]), 0);
                    val_0 = _mm_insert_epi32(val_0, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 0] + 1) + c)]), 1);
                    val_0 = _mm_insert_epi32(val_0, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 1]      + c)]), 2);
                    val_0 = _mm_insert_epi32(val_0, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 1] + 1) + c)]), 3);

                    val_1 = _mm_insert_epi32(val_1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 2]      + c)]), 0);
                    val_1 = _mm_insert_epi32(val_1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 2] + 1) + c)]), 1);
                    val_1 = _mm_insert_epi32(val_1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 3]      + c)]), 2);
                    val_1 = _mm_insert_epi32(val_1, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 3] + 1) + c)]), 3);

                    val_2 = _mm_insert_epi32(val_2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 4]      + c)]), 0);
                    val_2 = _mm_insert_epi32(val_2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 4] + 1) + c)]), 1);
                    val_2 = _mm_insert_epi32(val_2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 5]      + c)]), 2);
                    val_2 = _mm_insert_epi32(val_2, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 5] + 1) + c)]), 3);

                    val_3 = _mm_insert_epi32(val_3, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 6]      + c)]), 0);
                    val_3 = _mm_insert_epi32(val_3, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 6] + 1) + c)]), 1);
                    val_3 = _mm_insert_epi32(val_3, *reinterpret_cast<const int*>(&tmp[4 * (chanNum *  mapsx[x + 7]      + c)]), 2);
                    val_3 = _mm_insert_epi32(val_3, *reinterpret_cast<const int*>(&tmp[4 * (chanNum * (mapsx[x + 7] + 1) + c)]), 3);

                    val_0 = _mm_shuffle_epi32(val_0, _MM_SHUFFLE(3, 1, 2, 0));
                    val_1 = _mm_shuffle_epi32(val_1, _MM_SHUFFLE(3, 1, 2, 0));
                    val_2 = _mm_shuffle_epi32(val_2, _MM_SHUFFLE(3, 1, 2, 0));
                    val_3 = _mm_shuffle_epi32(val_3, _MM_SHUFFLE(3, 1, 2, 0));

                    __m128i val0_0 = _mm_cvtepu8_epi16(val_0);
                    __m128i val0_1 = _mm_cvtepu8_epi16(val_1);
                    __m128i val0_2 = _mm_cvtepu8_epi16(val_2);
                    __m128i val0_3 = _mm_cvtepu8_epi16(val_3);

                    __m128i val1_0 = _mm_unpackhi_epi8(val_0, _mm_setzero_si128());
                    __m128i val1_1 = _mm_unpackhi_epi8(val_1, _mm_setzero_si128());
                    __m128i val1_2 = _mm_unpackhi_epi8(val_2, _mm_setzero_si128());
                    __m128i val1_3 = _mm_unpackhi_epi8(val_3, _mm_setzero_si128());

                    __m128i t0 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_0, val1_0), a10);
                    __m128i t1 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_1, val1_1), a32);
                    __m128i t2 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_2, val1_2), a54);
                    __m128i t3 = _mm_mulhrs_epi16(_mm_sub_epi16(val0_3, val1_3), a76);

                    __m128i r0 = _mm_add_epi16(val1_0, t0);
                    __m128i r1 = _mm_add_epi16(val1_1, t1);
                    __m128i r2 = _mm_add_epi16(val1_2, t2);
                    __m128i r3 = _mm_add_epi16(val1_3, t3);

                    __m128i q0 = _mm_packus_epi16(r0, r1);
                    __m128i q1 = _mm_packus_epi16(r2, r3);

                    __m128i q2 = _mm_shuffle_epi8(q0, _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15));
                    __m128i q3 = _mm_shuffle_epi8(q1, _mm_setr_epi8(0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15));

                    __m128i q4 = _mm_blend_epi16(q2, _mm_slli_si128(q3, 4), 0xCC /*0b11001100*/);
                    __m128i q5 = _mm_blend_epi16(_mm_srli_si128(q2, 4), q3, 0xCC /*0b11001100*/);

                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[c][0][x]),                q4);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[c][1][x]), _mm_srli_si128(q4, 8));
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[c][2][x]),                q5);
                    _mm_storel_epi64(reinterpret_cast<__m128i*>(&dst[c][3][x]), _mm_srli_si128(q5, 8));
                }
            }

            if (x < outSz.width) {
                x = outSz.width - 8;
            }
        }
    } else {  // if any lpi
        for (int l = 0; l < lpi; l++) {
            short beta0 = beta[l];

            // vertical pass
            GAPI_DbgAssert(inSz.width*chanNum >= 8);
            for (int w = 0; w < inSz.width*chanNum; ) {
                for (; w <= inSz.width*chanNum - 8; w += 8) {
                    v_int16x8 s0 = v_reinterpret_as_s16(v_load_expand(&src0[l][w]));
                    v_int16x8 s1 = v_reinterpret_as_s16(v_load_expand(&src1[l][w]));
                    v_int16x8 t = v_mulhrs(s0 - s1, beta0) + s1;
                    v_pack_u_store(tmp + w, t);
                }

                if (w < inSz.width*chanNum) {
                    w = inSz.width*chanNum - 8;
                }
            }

            // horizontal pass
            GAPI_DbgAssert(outSz.width >= 8);
            for (int x = 0; x < outSz.width; ) {
                for (; x <= outSz.width - 8; x += 8) {
                    for (int c = 0; c < chanNum; c++) {
                        v_int16x8 a0 = v_load(&alpha[x]);        // as signed Q1.1.14
                        v_int16x8 sx = v_load(&mapsx[x]);        // as integer (int16)
                        v_int16x8 t0 = v_gather_chan(tmp, sx, c, 0);
                        v_int16x8 t1 = v_gather_chan(tmp, sx, c, 1);
                        v_int16x8 d = v_mulhrs(t0 - t1, a0) + t1;
                        v_pack_u_store(&dst[c][l][x], d);
                    }
                }

                if (x < outSz.width) {
                    x = outSz.width - 8;
                }
            }
        }
    }
}

// Resize (bi-linear, 32F)
void calcRowLinear_32F(float *dst[],
                 const float *src0[],
                 const float *src1[],
                 const float  alpha[],
                 const int    mapsx[],
                 const float  beta[],
                 const Size & inSz,
                 const Size & outSz,
                       int    lpi) {
    bool xRatioEq1 = inSz.width  == outSz.width;
    bool yRatioEq1 = inSz.height == outSz.height;

    if (!xRatioEq1 && !yRatioEq1) {
        for (int l = 0; l < lpi; l++) {
            float beta0 = beta[l];
            float beta1 = 1 - beta0;

            int x = 0;

        #if CV_SIMD128
            for (; x <= outSz.width - 4; x += 4) {
                v_float32x4 alpha0 = v_load(&alpha[x]);
            //  v_float32x4 alpha1 = 1.f - alpha0;

                v_int32x4 sx = v_load(&mapsx[x]);

                v_float32x4 s0l, s0h, s00, s01;
                v_gather_pairs(src0[l], sx, s0l, s0h);
                v_deinterleave(s0l, s0h, s00, s01);

            //  v_float32x4 res0 = s00*alpha0 + s01*alpha1;
                v_float32x4 res0 = v_fma(s00 - s01, alpha0, s01);

                v_float32x4 s1l, s1h, s10, s11;
                v_gather_pairs(src1[l], sx, s1l, s1h);
                v_deinterleave(s1l, s1h, s10, s11);

            //  v_float32x4 res1 = s10*alpha0 + s11*alpha1;
                v_float32x4 res1 = v_fma(s10 - s11, alpha0, s11);

            //  v_float32x4 d = res0*beta0 + res1*beta1;
                v_float32x4 d = v_fma(res0 - res1, beta0, res1);

                v_store(&dst[l][x], d);
            }
        #endif

            for (; x < outSz.width; x++) {
                float alpha0 = alpha[x];
                float alpha1 = 1 - alpha0;
                int   sx0 = mapsx[x];
                int   sx1 = sx0 + 1;
                float res0 = src0[l][sx0]*alpha0 + src0[l][sx1]*alpha1;
                float res1 = src1[l][sx0]*alpha0 + src1[l][sx1]*alpha1;
                dst[l][x] = beta0*res0 + beta1*res1;
            }
        }

    } else if (!xRatioEq1) {
        GAPI_DbgAssert(yRatioEq1);

        for (int l = 0; l < lpi; l++) {
            int x = 0;

        #if CV_SIMD128
            for (; x <= outSz.width - 4; x += 4) {
                v_float32x4 alpha0 = v_load(&alpha[x]);
            //  v_float32x4 alpha1 = 1.f - alpha0;

                v_int32x4 sx = v_load(&mapsx[x]);

                v_float32x4 s0l, s0h, s00, s01;
                v_gather_pairs(src0[l], sx, s0l, s0h);
                v_deinterleave(s0l, s0h, s00, s01);

            //  v_float32x4 d = s00*alpha0 + s01*alpha1;
                v_float32x4 d = v_fma(s00 - s01, alpha0, s01);

                v_store(&dst[l][x], d);
            }
        #endif

            for (; x < outSz.width; x++) {
                float alpha0 = alpha[x];
                float alpha1 = 1 - alpha0;
                int   sx0 = mapsx[x];
                int   sx1 = sx0 + 1;
                dst[l][x] = src0[l][sx0]*alpha0 + src0[l][sx1]*alpha1;
            }
        }

    } else if (!yRatioEq1) {
        GAPI_DbgAssert(xRatioEq1);
        int length = inSz.width;  // == outSz.width

        for (int l = 0; l < lpi; l++) {
            float beta0 = beta[l];
            float beta1 = 1 - beta0;

            int x = 0;

        #if CV_SIMD128
            for (; x <= length - 4; x += 4) {
                v_float32x4 s0 = v_load(&src0[l][x]);
                v_float32x4 s1 = v_load(&src1[l][x]);

            //  v_float32x4 d = s0*beta0 + s1*beta1;
                v_float32x4 d = v_fma(s0 - s1, beta0, s1);

                v_store(&dst[l][x], d);
            }
        #endif

            for (; x < length; x++) {
                dst[l][x] = beta0*src0[l][x] + beta1*src1[l][x];
            }
        }

    } else {
        GAPI_DbgAssert(xRatioEq1 && yRatioEq1);
        int length = inSz.width;  // == outSz.width
        for (int l = 0; l < lpi; l++) {
            memcpy(dst[l], src0[l], length * sizeof(float));
        }
    }
}

//------------------------------------------------------------------------------

// vertical pass
template<typename T, typename A, typename I, typename W>
static inline void downy(const T *src[], int inWidth, const MapperUnit<A, I>& ymap, A yalpha,
                         W vbuf[]) {
    int y_1st = ymap.index0;
    int ylast = ymap.index1 - 1;

    // yratio > 1, so at least 2 rows
    GAPI_DbgAssert(y_1st < ylast);

    // 1st and last rows
    {
        int w = 0;

    #if CV_SIMD128
        if (std::is_same<T, uint8_t>::value) {
            for (; w <= inWidth - 8; w += 8) {
                v_uint16x8 vsrc0 = v_load_expand(reinterpret_cast<const uint8_t*>(& src[0][w]));
                v_uint16x8 vsrc1 = v_load_expand(reinterpret_cast<const uint8_t*>(& src[ylast - y_1st][w]));
                v_uint16x8 vres = v_mulhi(vsrc0 << 8, static_cast<Q0_16>(ymap.alpha0)) +
                                  v_mulhi(vsrc1 << 8, static_cast<Q0_16>(ymap.alpha1));
                v_store(reinterpret_cast<Q8_8*>(& vbuf[w]), vres);
            }
        }
    #endif

        for (; w < inWidth; w++) {
            vbuf[w] = mulas(ymap.alpha0, src[0][w])
                    + mulas(ymap.alpha1, src[ylast - y_1st][w]);
        }
    }

    // inner rows (if any)
    for (int i = 1; i < ylast - y_1st; i++) {
        int w = 0;

    #if CV_SIMD128
        if (std::is_same<T, uint8_t>::value) {
            for (; w <= inWidth - 8; w += 8) {
                v_uint16x8 vsrc = v_load_expand(reinterpret_cast<const uint8_t*>(& src[i][w]));
                v_uint16x8 vres = v_load(reinterpret_cast<Q8_8*>(& vbuf[w]));
                vres = vres + v_mulhi(vsrc << 8, static_cast<Q0_16>(yalpha));
                v_store(reinterpret_cast<Q8_8*>(& vbuf[w]), vres);
            }
        }
    #endif

        for (; w < inWidth; w++) {
            vbuf[w] += mulas(yalpha, src[i][w]);
        }
    }
}

// horizontal pass
template<typename T, typename A, typename I, typename W>
static inline void downx(T dst[], int outWidth, int xmaxdf, const I xindex[], const A xalpha[],
                         const W vbuf[]) {
#define HSUM(xmaxdf) \
    for (int x = 0; x < outWidth; x++) { \
        int      index =  xindex[x]; \
        const A *alpha = &xalpha[x * xmaxdf]; \
\
        W sum = 0; \
        for (int i = 0; i < xmaxdf; i++) { \
            sum += mulaw(alpha[i], vbuf[index + i]); \
        } \
\
        dst[x] = convert_cast<T>(sum); \
    }

    if (2 == xmaxdf) {
        HSUM(2);
    } else if (3 == xmaxdf) {
        HSUM(3);
    } else if (4 == xmaxdf) {
        HSUM(4);
    } else if (5 == xmaxdf) {
        HSUM(5);
    } else if (6 == xmaxdf) {
        HSUM(6);
    } else if (7 == xmaxdf) {
        HSUM(7);
    } else if (8 == xmaxdf) {
        HSUM(8);
    } else {
        HSUM(xmaxdf);
    }
#undef HSUM
}

template<typename T, typename A, typename I, typename W>
static void calcRowArea_impl(T dst[], const T *src[], const Size& inSz, const Size& outSz,
    A yalpha, const MapperUnit<A, I>& ymap, int xmaxdf, const I xindex[], const A xalpha[],
    W vbuf[]) {
    bool xRatioEq1 = inSz.width  == outSz.width;
    bool yRatioEq1 = inSz.height == outSz.height;

    if (!yRatioEq1 && !xRatioEq1) {
        downy(src, inSz.width, ymap, yalpha, vbuf);
        downx(dst, outSz.width, xmaxdf, xindex, xalpha, vbuf);

    } else if (!yRatioEq1) {
        GAPI_DbgAssert(xRatioEq1);
        downy(src, inSz.width, ymap, yalpha, vbuf);
        for (int x = 0; x < outSz.width; x++) {
            dst[x] = convert_cast<T>(vbuf[x]);
        }

    } else if (!xRatioEq1) {
        GAPI_DbgAssert(yRatioEq1);
        for (int w = 0; w < inSz.width; w++) {
            vbuf[w] = convert_cast<W>(src[0][w]);
        }
        downx(dst, outSz.width, xmaxdf, xindex, xalpha, vbuf);

    } else {
        GAPI_DbgAssert(xRatioEq1 && yRatioEq1);
        memcpy(dst, src[0], outSz.width * sizeof(T));
    }
}

void calcRowArea_8U(uchar dst[], const uchar *src[], const Size& inSz, const Size& outSz,
    Q0_16 yalpha, const MapperUnit8U &ymap, int xmaxdf, const short xindex[], const Q0_16 xalpha[],
    Q8_8 vbuf[]) {
    calcRowArea_impl(dst, src, inSz, outSz, yalpha, ymap, xmaxdf, xindex, xalpha, vbuf);
}

void calcRowArea_32F(float dst[], const float *src[], const Size& inSz, const Size& outSz,
    float yalpha, const MapperUnit32F& ymap, int xmaxdf, const int xindex[], const float xalpha[],
    float vbuf[]) {
    calcRowArea_impl(dst, src, inSz, outSz, yalpha, ymap, xmaxdf, xindex, xalpha, vbuf);
}

//------------------------------------------------------------------------------
#if USE_CVKL

// from: ie_preprocess_data.hpp
static inline uint8_t saturateU32toU8(uint32_t v) {
    return static_cast<uint8_t>(v > UINT8_MAX ? UINT8_MAX : v);
}

// from: ie_preprocess_data_sse42.cpp
static inline uint16_t mulq16(uint16_t a, uint16_t b) {
    return static_cast<uint16_t>(((uint32_t)a * (uint32_t)b) >> 16);
}

// extracted from: ie_preprocess_data_sse42.cpp
// (and reworked for 1-channel and fluid's src)
void calcRowArea_CVKL_U8_SSE42(const uchar  * src[],
                                     uchar    dst[],
                               const Size   & inSz,
                               const Size   & outSz,
                                     int      y,
                               const uint16_t xsi[],
                               const uint16_t ysi[],
                               const uint16_t xalpha[],
                               const uint16_t yalpha[],
                                     int      x_max_count,
                                     int      y_max_count,
                                     uint16_t vert_sum[]) {
    int dwidth  = outSz.width;
//  int dheight = outSz.height;
    int swidth  =  inSz.width;
    int sheight =  inSz.height;

    int vest_sum_size = 2*swidth;
//  uint16_t* vert_sum = yalpha + dheight*y_max_count;
    uint16_t* alpha0 = vert_sum + vest_sum_size;
    uint16_t* alpha1 = alpha0 + dwidth;
    uint16_t* alpha2 = alpha1 + dwidth;
    uint16_t* alpha3 = alpha2 + dwidth;
    uint16_t* sxid0 = alpha3 + dwidth;
    uint16_t* sxid1 = sxid0 + 4*dwidth;
    uint16_t* sxid2 = sxid1 + 4*dwidth;
    uint16_t* sxid3 = sxid2 + 4*dwidth;

    uint8_t * pdst_row  = dst;
    uint16_t* vert_sum_ = vert_sum;

    int ysi_row = ysi[y];

    memset(vert_sum_, 0, swidth * sizeof(uint16_t));

    for (int dy = 0; dy < y_max_count; dy++) {
        if (ysi_row + dy >= sheight)
            break;

        uint16_t yalpha_dy = yalpha[y * y_max_count + dy];
        const uint8_t *sptr_dy = src[dy];

        int x = 0;

        __m128i yalpha_dy_sse = _mm_set1_epi16(yalpha_dy);
        for (; x <= swidth - 16; x += 16) {
            __m128i sval = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sptr_dy + x));

            // sptr_dy[x] << 8
            __m128i sval_Q16_lo = _mm_unpacklo_epi8(_mm_setzero_si128(), sval);
            __m128i sval_Q16_hi = _mm_unpackhi_epi8(_mm_setzero_si128(), sval);

            __m128i vert_sum_lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + x + 0));
            __m128i vert_sum_hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + x + 8));

            vert_sum_lo = _mm_add_epi16(vert_sum_lo, _mm_mulhi_epu16(yalpha_dy_sse, sval_Q16_lo));
            vert_sum_hi = _mm_add_epi16(vert_sum_hi, _mm_mulhi_epu16(yalpha_dy_sse, sval_Q16_hi));

            _mm_storeu_si128(reinterpret_cast<__m128i*>(vert_sum_ + x + 0), vert_sum_lo);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(vert_sum_ + x + 8), vert_sum_hi);
        }

        for (; x < swidth; x++) {
            vert_sum_[x] += mulq16(yalpha_dy, static_cast<uint16_t>(sptr_dy[x] << 8));
        }
    }

    if (x_max_count == 2) {
        int x = 0;
        for (; x <= dwidth - 8; x += 8) {
            __m128i res = _mm_set1_epi16(1 << (8 - 1));

            int id0 = xsi[x];

            __m128i chunk0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0));
            __m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 8));

            __m128i sx0_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 2));
            __m128i sx0_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 2 + 8));

            __m128i sx1_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 2));
            __m128i sx1_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 2 + 8));

            __m128i vert_sum0 = _mm_or_si128(_mm_shuffle_epi8(chunk0, sx0_id0),
                                             _mm_shuffle_epi8(chunk1, sx0_id1));
            __m128i vert_sum1 = _mm_or_si128(_mm_shuffle_epi8(chunk0, sx1_id0),
                                             _mm_shuffle_epi8(chunk1, sx1_id1));

            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha0 + x)), vert_sum0));
            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha1 + x)), vert_sum1));

            res = _mm_srli_epi16(res, 8);
            res = _mm_packus_epi16(res, res);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
        }

        for (; x < dwidth; x++) {
            uint16_t res = 1 << (8 - 1);
            int id = xsi[x];
            res += mulq16(alpha0[x], vert_sum_[id + 0]);
            res += mulq16(alpha1[x], vert_sum_[id + 1]);
            pdst_row[x] = saturateU32toU8(res >> 8);
        }
    } else if (x_max_count == 3) {
        int x = 0;
        for (; x <= dwidth - 8; x += 8) {
            __m128i res = _mm_set1_epi16(1 << (8 - 1));

            int id0 = xsi[x];

            __m128i chunk0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0));
            __m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 8));
            __m128i chunk2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 16));

            __m128i sx0_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 3));
            __m128i sx0_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 3 + 8));
            __m128i sx0_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 3 + 16));

            __m128i sx1_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 3));
            __m128i sx1_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 3 + 8));
            __m128i sx1_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 3 + 16));

            __m128i sx2_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 3));
            __m128i sx2_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 3 + 8));
            __m128i sx2_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 3 + 16));

            __m128i vert_sum0 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx0_id0),
                                                          _mm_shuffle_epi8(chunk1, sx0_id1)),
                                             _mm_shuffle_epi8(chunk2, sx0_id2));
            __m128i vert_sum1 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx1_id0),
                                                          _mm_shuffle_epi8(chunk1, sx1_id1)),
                                             _mm_shuffle_epi8(chunk2, sx1_id2));
            __m128i vert_sum2 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx2_id0),
                                                          _mm_shuffle_epi8(chunk1, sx2_id1)),
                                             _mm_shuffle_epi8(chunk2, sx2_id2));

            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha0 + x)), vert_sum0));
            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha1 + x)), vert_sum1));
            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha2 + x)), vert_sum2));

            res = _mm_srli_epi16(res, 8);
            res = _mm_packus_epi16(res, res);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
        }

        for (; x < dwidth; x++) {
            uint16_t res = 1 << (8 - 1);
            int id = xsi[x];
            res += mulq16(alpha0[x], vert_sum_[id + 0]);
            res += mulq16(alpha1[x], vert_sum_[id + 1]);
            res += mulq16(alpha2[x], vert_sum_[id + 2]);
            pdst_row[x] = saturateU32toU8(res >> 8);
        }
    } else if (x_max_count == 4) {
        int x = 0;
        for (; x <= dwidth - 8; x += 8) {
            __m128i res = _mm_set1_epi16(1 << (8 - 1));

            int id0 = xsi[x];

            __m128i chunk0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0));
            __m128i chunk1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 8));
            __m128i chunk2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 16));
            __m128i chunk3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id0 + 24));

            __m128i sx0_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4));
            __m128i sx0_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4 + 8));
            __m128i sx0_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4 + 16));
            __m128i sx0_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid0 + x * 4 + 24));

            __m128i sx1_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4));
            __m128i sx1_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4 + 8));
            __m128i sx1_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4 + 16));
            __m128i sx1_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid1 + x * 4 + 24));

            __m128i sx2_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4));
            __m128i sx2_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4 + 8));
            __m128i sx2_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4 + 16));
            __m128i sx2_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid2 + x * 4 + 24));

            __m128i sx3_id0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4));
            __m128i sx3_id1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4 + 8));
            __m128i sx3_id2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4 + 16));
            __m128i sx3_id3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sxid3 + x * 4 + 24));

            __m128i vert_sum0 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx0_id0),
                                                          _mm_shuffle_epi8(chunk1, sx0_id1)),
                                             _mm_or_si128(_mm_shuffle_epi8(chunk2, sx0_id2),
                                                          _mm_shuffle_epi8(chunk3, sx0_id3)));
            __m128i vert_sum1 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx1_id0),
                                                          _mm_shuffle_epi8(chunk1, sx1_id1)),
                                             _mm_or_si128(_mm_shuffle_epi8(chunk2, sx1_id2),
                                                          _mm_shuffle_epi8(chunk3, sx1_id3)));
            __m128i vert_sum2 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx2_id0),
                                                          _mm_shuffle_epi8(chunk1, sx2_id1)),
                                             _mm_or_si128(_mm_shuffle_epi8(chunk2, sx2_id2),
                                                          _mm_shuffle_epi8(chunk3, sx2_id3)));
            __m128i vert_sum3 = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(chunk0, sx3_id0),
                                                          _mm_shuffle_epi8(chunk1, sx3_id1)),
                                             _mm_or_si128(_mm_shuffle_epi8(chunk2, sx3_id2),
                                                          _mm_shuffle_epi8(chunk3, sx3_id3)));

            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha0 + x)), vert_sum0));
            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha1 + x)), vert_sum1));
            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha2 + x)), vert_sum2));
            res = _mm_add_epi16(res, _mm_mulhi_epu16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(alpha3 + x)), vert_sum3));

            res = _mm_srli_epi16(res, 8);
            res = _mm_packus_epi16(res, res);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
        }

        for (; x < dwidth; x++) {
            uint16_t res = 1 << (8 - 1);
            int id = xsi[x];
            res += mulq16(alpha0[x], vert_sum_[id + 0]);
            res += mulq16(alpha1[x], vert_sum_[id + 1]);
            res += mulq16(alpha2[x], vert_sum_[id + 2]);
            res += mulq16(alpha3[x], vert_sum_[id + 3]);
            pdst_row[x] = saturateU32toU8(res >> 8);
        }
    } else if (x_max_count <= 7) {
        int x = 0;
        for (; x <= dwidth - 8; x += 8) {
            __m128i res = _mm_set1_epi16(1 << (16 - 8 - 1));
            for (int i = 0; i < x_max_count; i++) {
                __m128i valpha = _mm_setr_epi16(xalpha[x * x_max_count + x_max_count * 0 + i],
                                                xalpha[x * x_max_count + x_max_count * 1 + i],
                                                xalpha[x * x_max_count + x_max_count * 2 + i],
                                                xalpha[x * x_max_count + x_max_count * 3 + i],
                                                xalpha[x * x_max_count + x_max_count * 4 + i],
                                                xalpha[x * x_max_count + x_max_count * 5 + i],
                                                xalpha[x * x_max_count + x_max_count * 6 + i],
                                                xalpha[x * x_max_count + x_max_count * 7 + i]);
                __m128i vvert_sum = _mm_setr_epi16(vert_sum_[xsi[x + 0] + i],
                                                   vert_sum_[xsi[x + 1] + i],
                                                   vert_sum_[xsi[x + 2] + i],
                                                   vert_sum_[xsi[x + 3] + i],
                                                   vert_sum_[xsi[x + 4] + i],
                                                   vert_sum_[xsi[x + 5] + i],
                                                   vert_sum_[xsi[x + 6] + i],
                                                   vert_sum_[xsi[x + 7] + i]);

                res = _mm_add_epi16(res, _mm_mulhi_epu16(valpha, vvert_sum));
            }
            res = _mm_srli_epi16(res, 8);
            res = _mm_packus_epi16(res, res);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(pdst_row + x), res);
        }

        for (; x < dwidth; x++) {
            uint16_t res = 1 << (8 - 1);
            for (int i = 0; i < x_max_count; i++) {
                uint16_t a = xalpha[x * x_max_count + i];
                int sx = xsi[x] + i;

                res += mulq16(a, vert_sum_[sx]);
            }
            pdst_row[x] = saturateU32toU8(res >> 8);
        }
    } else {
        for (int x = 0; x < dwidth; x++) {
            uint16_t res = 1 << (8 - 1);
            __m128i vres = _mm_setzero_si128();
            int id = xsi[x];

            int i = 0;
            for (; i <= x_max_count - 8; i += 8) {
                __m128i a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(xalpha + x * x_max_count + i));
                __m128i s = _mm_loadu_si128(reinterpret_cast<const __m128i*>(vert_sum_ + id + i));

                vres = _mm_add_epi16(vres, _mm_mulhi_epu16(a, s));
            }
            vres = _mm_add_epi16(vres, _mm_slli_si128(vres, 2));
            vres = _mm_add_epi16(vres, _mm_slli_si128(vres, 4));
            vres = _mm_add_epi16(vres, _mm_slli_si128(vres, 8));
            res += static_cast<uint16_t>(_mm_extract_epi16(vres, 7));

            for (; i < x_max_count; i++) {
                uint16_t a = xalpha[x * x_max_count + i];
                uint16_t s = vert_sum_[id + i];

                res += mulq16(a, s);
            }

            pdst_row[x] = saturateU32toU8(res >> 8);
        }
    }
}

#endif  // CVKL
//------------------------------------------------------------------------------

void mergeRow_8UC2(const uint8_t in0[],
                   const uint8_t in1[],
                         uint8_t out[],
                             int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 16; l += 16) {
        v_uint8x16 r0, r1;
        r0 = v_load(&in0[l]);
        r1 = v_load(&in1[l]);
        v_store_interleave(&out[2*l], r0, r1);
    }

    // FIXME: get rid of all gotos below
    // Also to think about how to remove those ifs
    if (l < length && length >= 16) {
        l = length - 16;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out[2*l + 0] = in0[l];
        out[2*l + 1] = in1[l];
    }
}

void mergeRow_8UC3(const uint8_t in0[],
                   const uint8_t in1[],
                   const uint8_t in2[],
                         uint8_t out[],
                             int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 16; l += 16) {
        v_uint8x16 r0, r1, r2;
        r0 = v_load(&in0[l]);
        r1 = v_load(&in1[l]);
        r2 = v_load(&in2[l]);
        v_store_interleave(&out[3*l], r0, r1, r2);
    }

    if (l < length && length >= 16) {
        l = length - 16;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out[3*l + 0] = in0[l];
        out[3*l + 1] = in1[l];
        out[3*l + 2] = in2[l];
    }
}

void mergeRow_8UC4(const uint8_t in0[],
                   const uint8_t in1[],
                   const uint8_t in2[],
                   const uint8_t in3[],
                         uint8_t out[],
                             int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 16; l += 16) {
        v_uint8x16 r0, r1, r2, r3;
        r0 = v_load(&in0[l]);
        r1 = v_load(&in1[l]);
        r2 = v_load(&in2[l]);
        r3 = v_load(&in3[l]);
        v_store_interleave(&out[4*l], r0, r1, r2, r3);
    }

    if (l < length && length >= 16) {
        l = length - 16;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out[4*l + 0] = in0[l];
        out[4*l + 1] = in1[l];
        out[4*l + 2] = in2[l];
        out[4*l + 3] = in3[l];
    }
}

void mergeRow_32FC2(const float in0[],
                    const float in1[],
                          float out[],
                            int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 4; l += 4) {
        v_float32x4 r0, r1;
        r0 = v_load(&in0[l]);
        r1 = v_load(&in1[l]);
        v_store_interleave(&out[2*l], r0, r1);
    }

    if (l < length && length >= 4) {
        l = length - 4;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out[2*l + 0] = in0[l];
        out[2*l + 1] = in1[l];
    }
}

void mergeRow_32FC3(const float in0[],
                    const float in1[],
                    const float in2[],
                          float out[],
                            int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 4; l += 4) {
        v_float32x4 r0, r1, r2;
        r0 = v_load(&in0[l]);
        r1 = v_load(&in1[l]);
        r2 = v_load(&in2[l]);
        v_store_interleave(&out[3*l], r0, r1, r2);
    }

    if (l < length && length >= 4) {
        l = length - 4;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out[3*l + 0] = in0[l];
        out[3*l + 1] = in1[l];
        out[3*l + 2] = in2[l];
    }
}

void mergeRow_32FC4(const float in0[],
                    const float in1[],
                    const float in2[],
                    const float in3[],
                          float out[],
                            int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 4; l += 4) {
        v_float32x4 r0, r1, r2, r3;
        r0 = v_load(&in0[l]);
        r1 = v_load(&in1[l]);
        r2 = v_load(&in2[l]);
        r3 = v_load(&in3[l]);
        v_store_interleave(&out[4*l], r0, r1, r2, r3);
    }

    if (l < length && length >= 4) {
        l = length - 4;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out[4*l + 0] = in0[l];
        out[4*l + 1] = in1[l];
        out[4*l + 2] = in2[l];
        out[4*l + 3] = in3[l];
    }
}

void splitRow_8UC2(const uint8_t in[],
                         uint8_t out0[],
                         uint8_t out1[],
                             int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 16; l += 16) {
        v_uint8x16 r0, r1;
        v_load_deinterleave(&in[2*l], r0, r1);
        v_store(&out0[l], r0);
        v_store(&out1[l], r1);
    }
    if (l < length && length >= 16) {
        l = length - 16;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out0[l] = in[2*l + 0];
        out1[l] = in[2*l + 1];
    }
}

void splitRow_8UC3(const uint8_t in[],
                         uint8_t out0[],
                         uint8_t out1[],
                         uint8_t out2[],
                             int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 16; l += 16) {
        v_uint8x16 r0, r1, r2;
        v_load_deinterleave(&in[3*l], r0, r1, r2);
        v_store(&out0[l], r0);
        v_store(&out1[l], r1);
        v_store(&out2[l], r2);
    }
    if (l < length && length >= 16) {
        l = length - 16;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out0[l] = in[3*l + 0];
        out1[l] = in[3*l + 1];
        out2[l] = in[3*l + 2];
    }
}

void splitRow_8UC4(const uint8_t in[],
                         uint8_t out0[],
                         uint8_t out1[],
                         uint8_t out2[],
                         uint8_t out3[],
                             int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 16; l += 16) {
        v_uint8x16 r0, r1, r2, r3;
        v_load_deinterleave(&in[4*l], r0, r1, r2, r3);
        v_store(&out0[l], r0);
        v_store(&out1[l], r1);
        v_store(&out2[l], r2);
        v_store(&out3[l], r3);
    }
    if (l < length && length >= 16) {
        l = length - 16;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out0[l] = in[4*l + 0];
        out1[l] = in[4*l + 1];
        out2[l] = in[4*l + 2];
        out3[l] = in[4*l + 3];
    }
}

void splitRow_32FC2(const float in[],
                          float out0[],
                          float out1[],
                            int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 4; l += 4) {
        v_float32x4 r0, r1;
        v_load_deinterleave(&in[2*l], r0, r1);
        v_store(&out0[l], r0);
        v_store(&out1[l], r1);
    }

    if (l < length && length >= 4) {
        l = length - 4;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out0[l] = in[2*l + 0];
        out1[l] = in[2*l + 1];
    }
}

void splitRow_32FC3(const float in[],
                          float out0[],
                          float out1[],
                          float out2[],
                            int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 4; l += 4) {
        v_float32x4 r0, r1, r2;
        v_load_deinterleave(&in[3*l], r0, r1, r2);
        v_store(&out0[l], r0);
        v_store(&out1[l], r1);
        v_store(&out2[l], r2);
    }

    if (l < length && length >= 4) {
        l = length - 4;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out0[l] = in[3*l + 0];
        out1[l] = in[3*l + 1];
        out2[l] = in[3*l + 2];
    }
}

void splitRow_32FC4(const float in[],
                          float out0[],
                          float out1[],
                          float out2[],
                          float out3[],
                            int length) {
    int l = 0;

#if CV_SIMD128
    cycle:
    for (; l <= length - 4; l += 4) {
        v_float32x4 r0, r1, r2, r3;
        v_load_deinterleave(&in[4*l], r0, r1, r2, r3);
        v_store(&out0[l], r0);
        v_store(&out1[l], r1);
        v_store(&out2[l], r2);
        v_store(&out3[l], r3);
    }

    if (l < length && length >= 4) {
        l = length - 4;
        goto cycle;
    }
#endif

    for (; l < length; l++) {
        out0[l] = in[4*l + 0];
        out1[l] = in[4*l + 1];
        out2[l] = in[4*l + 2];
        out3[l] = in[4*l + 3];
    }
}

static const int ITUR_BT_601_CY = 1220542;
static const int ITUR_BT_601_CUB = 2116026;
static const int ITUR_BT_601_CUG = -409993;
static const int ITUR_BT_601_CVG = -852492;
static const int ITUR_BT_601_CVR = 1673527;
static const int ITUR_BT_601_SHIFT = 20;

static inline void uvToRGBuv(const uchar u, const uchar v, int& ruv, int& guv, int& buv) {
    int uu, vv;
    uu = static_cast<int>(u) - 128;
    vv = static_cast<int>(v) - 128;

    ruv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVR * vv;
    guv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CVG * vv + ITUR_BT_601_CUG * uu;
    buv = (1 << (ITUR_BT_601_SHIFT - 1)) + ITUR_BT_601_CUB * uu;
}

static inline void uvToRGBuv(const v_uint8x16& u, const v_uint8x16& v,
                             v_int32x4 (&ruv)[4],
                             v_int32x4 (&guv)[4],
                             v_int32x4 (&buv)[4]) {
    v_uint8x16 v128 = v_setall_u8(128);
    v_int8x16 su = v_reinterpret_as_s8(v_sub_wrap(u, v128));
    v_int8x16 sv = v_reinterpret_as_s8(v_sub_wrap(v, v128));

    v_int16x8 uu0, uu1, vv0, vv1;
    v_expand(su, uu0, uu1);
    v_expand(sv, vv0, vv1);
    v_int32x4 uu[4], vv[4];
    v_expand(uu0, uu[0], uu[1]); v_expand(uu1, uu[2], uu[3]);
    v_expand(vv0, vv[0], vv[1]); v_expand(vv1, vv[2], vv[3]);

    v_int32x4 vshift = v_setall_s32(1 << (ITUR_BT_601_SHIFT - 1));
    v_int32x4 vr = v_setall_s32(ITUR_BT_601_CVR);
    v_int32x4 vg = v_setall_s32(ITUR_BT_601_CVG);
    v_int32x4 ug = v_setall_s32(ITUR_BT_601_CUG);
    v_int32x4 ub = v_setall_s32(ITUR_BT_601_CUB);

    for (int k = 0; k < 4; k++) {
        ruv[k] = vshift + vr * vv[k];
        guv[k] = vshift + vg * vv[k] + ug * uu[k];
        buv[k] = vshift + ub * uu[k];
    }
}

static inline void yRGBuvToRGB(const uchar vy, const int ruv, const int guv, const int buv,
                                uchar& r, uchar& g, uchar& b) {
    int yy = static_cast<int>(vy);
    int y = std::max(0, yy - 16) * ITUR_BT_601_CY;
    r = saturate_cast<uchar>((y + ruv) >> ITUR_BT_601_SHIFT);
    g = saturate_cast<uchar>((y + guv) >> ITUR_BT_601_SHIFT);
    b = saturate_cast<uchar>((y + buv) >> ITUR_BT_601_SHIFT);
}


static inline void yRGBuvToRGB(const v_uint8x16& vy,
                                const v_int32x4 (&ruv)[4],
                                const v_int32x4 (&guv)[4],
                                const v_int32x4 (&buv)[4],
                                v_uint8x16& rr, v_uint8x16& gg, v_uint8x16& bb) {
    v_uint8x16 v16 = v_setall_u8(16);
    v_uint8x16 posY = vy - v16;
    v_uint16x8 yy0, yy1;
    v_expand(posY, yy0, yy1);
    v_int32x4 yy[4];
    v_int32x4 yy00, yy01, yy10, yy11;
    v_expand(v_reinterpret_as_s16(yy0), yy[0], yy[1]);
    v_expand(v_reinterpret_as_s16(yy1), yy[2], yy[3]);

    v_int32x4 vcy = v_setall_s32(ITUR_BT_601_CY);

    v_int32x4 y[4], r[4], g[4], b[4];
    for (int k = 0; k < 4; k++) {
        y[k] = yy[k]*vcy;
        r[k] = (y[k] + ruv[k]) >> ITUR_BT_601_SHIFT;
        g[k] = (y[k] + guv[k]) >> ITUR_BT_601_SHIFT;
        b[k] = (y[k] + buv[k]) >> ITUR_BT_601_SHIFT;
    }

    v_int16x8 r0, r1, g0, g1, b0, b1;
    r0 = v_pack(r[0], r[1]);
    r1 = v_pack(r[2], r[3]);
    g0 = v_pack(g[0], g[1]);
    g1 = v_pack(g[2], g[3]);
    b0 = v_pack(b[0], b[1]);
    b1 = v_pack(b[2], b[3]);

    rr = v_pack_u(r0, r1);
    gg = v_pack_u(g0, g1);
    bb = v_pack_u(b0, b1);
}

void calculate_nv12_to_rgb(const  uchar **srcY,
                           const  uchar *srcUV,
                                  uchar **dstRGBx,
                                    int width) {
    int i = 0;

    #if CV_SIMD128

    const int vsize = v_uint8x16::nlanes;

    for ( ; i <= width - 2*vsize; i += 2*vsize) {
        v_uint8x16 u, v;
        v_load_deinterleave(srcUV + i, u, v);

        v_uint8x16 vy[4];
        v_load_deinterleave(srcY[0] + i, vy[0], vy[1]);
        v_load_deinterleave(srcY[1] + i, vy[2], vy[3]);

        v_int32x4 ruv[4], guv[4], buv[4];
        uvToRGBuv(u, v, ruv, guv, buv);

        v_uint8x16 r[4], g[4], b[4];

        for (int k = 0; k < 4; k++) {
            yRGBuvToRGB(vy[k], ruv, guv, buv, r[k], g[k], b[k]);
        }

        for (int k = 0; k < 4; k++)
            std::swap(r[k], b[k]);

        // [r0...], [r1...] => [r0, r1, r0, r1...], [r0, r1, r0, r1...]
        v_uint8x16 r0_0, r0_1, r1_0, r1_1;
        v_zip(r[0], r[1], r0_0, r0_1);
        v_zip(r[2], r[3], r1_0, r1_1);
        v_uint8x16 g0_0, g0_1, g1_0, g1_1;
        v_zip(g[0], g[1], g0_0, g0_1);
        v_zip(g[2], g[3], g1_0, g1_1);
        v_uint8x16 b0_0, b0_1, b1_0, b1_1;
        v_zip(b[0], b[1], b0_0, b0_1);
        v_zip(b[2], b[3], b1_0, b1_1);

        v_store_interleave(dstRGBx[0] + i * 3, b0_0, g0_0, r0_0);
        v_store_interleave(dstRGBx[0] + i * 3 + 3 * vsize, b0_1, g0_1, r0_1);

        v_store_interleave(dstRGBx[1] + i * 3, b1_0, g1_0, r1_0);
        v_store_interleave(dstRGBx[1] + i * 3 + 3 * vsize, b1_1, g1_1, r1_1);
    }

    v_cleanup();

    #endif

    for (; i < width; i += 2) {
        uchar u = srcUV[i];
        uchar v = srcUV[i + 1];
        int ruv, guv, buv;
        uvToRGBuv(u, v, ruv, guv, buv);

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                uchar vy = srcY[y][i + x];
                uchar r, g, b;
                yRGBuvToRGB(vy, ruv, guv, buv, r, g, b);

                dstRGBx[y][3*(i + x)]     = r;
                dstRGBx[y][3*(i + x) + 1] = g;
                dstRGBx[y][3*(i + x) + 2] = b;
            }
        }
    }
}

template <typename VecT, typename T>
void copyRow_impl(const T in[], T out[], int l) {
    VecT r;
    r = v_load(&in[l]);
    v_store(&out[l], r);
}

void copyRow_8U(const uint8_t in[],
                 uint8_t out[],
                 int length) {
    int l = 0;

#if CV_SIMD128
    for (; l <= length - 16; l += 16) {
        copyRow_impl<v_uint8x16>(in, out, l);
    }

    if (l < length && length >= 16) {
        copyRow_impl<v_uint8x16>(in, out, length - 16);
        l = length;
    }
#endif

    for (; l < length; l++) {
        out[l] = in[l];
    }
}

void copyRow_32F(const float in[],
                 float out[],
                 int length) {
    int l = 0;

#if CV_SIMD128
    for (; l <= length - 4; l += 4) {
        copyRow_impl<v_float32x4>(in, out, l);
    }

    if (l < length && length >= 4) {
        copyRow_impl<v_float32x4>(in, out, length - 4);
        l = length;
    }
#endif

    for (; l < length; l++) {
        out[l] = in[l];
    }
}

}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine
