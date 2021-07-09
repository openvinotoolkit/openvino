// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef IE_PREPROCESS_GAPI_KERNELS_SIMD_IMPL_H
#define IE_PREPROCESS_GAPI_KERNELS_SIMD_IMPL_H

#include <algorithm>
#include <utility>

#include "ie_preprocess_gapi_kernels_impl.hpp"

using namespace cv;

namespace InferenceEngine {

namespace gapi {

namespace kernels {

template <typename VecT, typename T>
CV_ALWAYS_INLINE void mergeRowC2_Impl(const T in0[], const T in1[],
                                      T out[], const int length) {
    int x = 0;

#if MANUAL_SIMD
    constexpr int nlanes = VecT::nlanes;
    GAPI_DbgAssert(length >= nlanes);

    VecT r0, r1;
    for (; length >= nlanes;) {
        for (; x <= length - nlanes; x += nlanes) {
            r0 = vx_load(&in0[x]);
            r1 = vx_load(&in1[x]);
            v_store_interleave(&out[2*x], r0, r1);
        }

        if (x < length) {
            x = length - nlanes;
            continue;
        }
        break;
    }
#endif

    for (; x < length; ++x) {
        out[2*x + 0] = in0[x];
        out[2*x + 1] = in1[x];
    }
}

template <typename VecT, typename T>
CV_ALWAYS_INLINE void mergeRowC3_Impl(const T in0[], const T in1[],
                                      const T in2[], T out[], const int length) {
    int x = 0;

#if MANUAL_SIMD
    constexpr int nlanes = VecT::nlanes;
    GAPI_DbgAssert(length >= nlanes);

    VecT r0, r1, r2;
    for (; length >= nlanes;) {
        for (; x <= length - nlanes; x += nlanes) {
            r0 = vx_load(&in0[x]);
            r1 = vx_load(&in1[x]);
            r2 = vx_load(&in2[x]);
            v_store_interleave(&out[3*x], r0, r1, r2);
        }

        if (x < length) {
            x = length - nlanes;
            continue;
        }
        break;
    }
#endif

    for (; x < length; ++x) {
        out[3*x + 0] = in0[x];
        out[3*x + 1] = in1[x];
        out[3*x + 2] = in2[x];
    }
}

template <typename VecT, typename T>
CV_ALWAYS_INLINE void mergeRowC4_Impl(const T in0[], const T in1[],
                                      const T in2[], const T in3[],
                                      T out[], const int length) {
    int x = 0;

#if MANUAL_SIMD
    constexpr int nlanes = VecT::nlanes;
    GAPI_DbgAssert(length >= nlanes);

    VecT r0, r1, r2, r3;
    for (; length >= nlanes;) {
        for (; x <= length - nlanes; x += nlanes) {
            r0 = vx_load(&in0[x]);
            r1 = vx_load(&in1[x]);
            r2 = vx_load(&in2[x]);
            r3 = vx_load(&in3[x]);
            v_store_interleave(&out[4* x], r0, r1, r2, r3);
        }

        if (x < length) {
            x = length - nlanes;
            continue;
        }
        break;
    }
#endif

    for (; x < length; ++x) {
        out[4*x + 0] = in0[x];
        out[4*x + 1] = in1[x];
        out[4*x + 2] = in2[x];
        out[4*x + 3] = in3[x];
    }
}
//------------------------------------------------------------------------------
template <typename VecT, typename T>
CV_ALWAYS_INLINE void splitRowC2_Impl(const T in[], T out0[],
                                      T out1[], const int length) {
    int x = 0;

#if MANUAL_SIMD
    constexpr int nlanes = VecT::nlanes;
    GAPI_DbgAssert(length >= nlanes);

    VecT r0, r1;
    for (; length >= nlanes;) {
        for (; x <= length - nlanes; x += nlanes) {
            v_load_deinterleave(&in[2*x], r0, r1);
            vx_store(&out0[x], r0);
            vx_store(&out1[x], r1);
        }

        if (x < length) {
            x = length - nlanes;
            continue;
        }
        break;
    }
#endif

    for (; x < length; ++x) {
        out0[x] = in[2*x + 0];
        out1[x] = in[2*x + 1];
    }
}

template <typename VecT, typename T>
CV_ALWAYS_INLINE void splitRowC3_Impl(const T in[], T out0[],
                                      T out1[], T out2[], const int length) {
    int x = 0;

#if MANUAL_SIMD
    constexpr int nlanes = VecT::nlanes;
    GAPI_DbgAssert(length >= nlanes);

    VecT r0, r1, r2;
    for (; length >= nlanes;) {
        for (; x <= length - nlanes; x += nlanes) {
             v_load_deinterleave(&in[3*x], r0, r1, r2);
             vx_store(&out0[x], r0);
             vx_store(&out1[x], r1);
             vx_store(&out2[x], r2);
        }

        if (x < length) {
            x = length - nlanes;
            continue;
        }
        break;
    }
#endif

    for (; x < length; ++x) {
        out0[x] = in[3*x + 0];
        out1[x] = in[3*x + 1];
        out2[x] = in[3*x + 2];
    }
}

template <typename VecT, typename T>
CV_ALWAYS_INLINE void splitRowC4_Impl(const T in[], T out0[], T out1[],
                                      T out2[], T out3[], const int length) {
    int x = 0;

#if MANUAL_SIMD
    constexpr int nlanes = VecT::nlanes;
    GAPI_DbgAssert(length >= nlanes);

    VecT r0, r1, r2, r3;
    for (; length >= nlanes;) {
        for (; x <= length - nlanes; x += nlanes) {
            v_load_deinterleave(&in[4*x], r0, r1, r2, r3);
            vx_store(&out0[x], r0);
            vx_store(&out1[x], r1);
            vx_store(&out2[x], r2);
            vx_store(&out3[x], r3);
        }

        if (x < length) {
            x = length - nlanes;
            continue;
        }
        break;
    }
#endif

    for (; x < length; ++x) {
        out0[x] = in[4*x + 0];
        out1[x] = in[4*x + 1];
        out2[x] = in[4*x + 2];
        out3[x] = in[4*x + 3];
    }
}
//------------------------------------------------------------------------------

CV_ALWAYS_INLINE void uvToRGBuv(const v_uint8& u, const v_uint8& v,
                                v_int32 (&ruv)[4], v_int32 (&guv)[4],
                                v_int32 (&buv)[4]) {
    v_uint8 v128 = vx_setall_u8(128);
    v_int8 su = v_reinterpret_as_s8(v_sub_wrap(u, v128));
    v_int8 sv = v_reinterpret_as_s8(v_sub_wrap(v, v128));

    v_int16 uu0, uu1, vv0, vv1;
    v_expand(su, uu0, uu1);
    v_expand(sv, vv0, vv1);
    v_int32 uu[4], vv[4];
    v_expand(uu0, uu[0], uu[1]); v_expand(uu1, uu[2], uu[3]);
    v_expand(vv0, vv[0], vv[1]); v_expand(vv1, vv[2], vv[3]);

    v_int32 vshift = vx_setall_s32(1 << (ITUR_BT_601_SHIFT - 1));
    v_int32 vr = vx_setall_s32(ITUR_BT_601_CVR);
    v_int32 vg = vx_setall_s32(ITUR_BT_601_CVG);
    v_int32 ug = vx_setall_s32(ITUR_BT_601_CUG);
    v_int32 ub = vx_setall_s32(ITUR_BT_601_CUB);

    for (int k = 0; k < 4; k++) {
        ruv[k] = vshift + vr * vv[k];
        guv[k] = vshift + vg * vv[k] + ug * uu[k];
        buv[k] = vshift + ub * uu[k];
    }
}

CV_ALWAYS_INLINE void yRGBuvToRGB(const v_uint8& vy,
                                  const v_int32 (&ruv)[4],
                                  const v_int32 (&guv)[4],
                                  const v_int32 (&buv)[4],
                                  v_uint8& rr, v_uint8& gg, v_uint8& bb) {
    v_uint8 v16 = vx_setall_u8(16);
    v_uint8 posY = vy - v16;
    v_uint16 yy0, yy1;
    v_expand(posY, yy0, yy1);
    v_int32 yy[4];
    v_int32 yy00, yy01, yy10, yy11;
    v_expand(v_reinterpret_as_s16(yy0), yy[0], yy[1]);
    v_expand(v_reinterpret_as_s16(yy1), yy[2], yy[3]);

    v_int32 vcy = vx_setall_s32(ITUR_BT_601_CY);

    v_int32 y[4], r[4], g[4], b[4];
    for (int k = 0; k < 4; k++) {
        y[k] = yy[k]*vcy;
        r[k] = (y[k] + ruv[k]) >> ITUR_BT_601_SHIFT;
        g[k] = (y[k] + guv[k]) >> ITUR_BT_601_SHIFT;
        b[k] = (y[k] + buv[k]) >> ITUR_BT_601_SHIFT;
    }

    v_int16 r0, r1, g0, g1, b0, b1;
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

template<typename isa_tag_t>
CV_ALWAYS_INLINE void nv12ToRgbRowImpl(isa_tag_t, const uchar** srcY, const uchar* srcUV,
                                       uchar** dstRGBx, const int width) {
    int i = 0;

#if MANUAL_SIMD
    constexpr int nlanes = v_uint8::nlanes;

    for (; i <= width - 2 * nlanes; i += 2 * nlanes) {
        v_uint8 u, v;
        v_load_deinterleave(srcUV + i, u, v);

        v_uint8 vy[4];
        v_load_deinterleave(srcY[0] + i, vy[0], vy[1]);
        v_load_deinterleave(srcY[1] + i, vy[2], vy[3]);

        v_int32 ruv[4], guv[4], buv[4];
        uvToRGBuv(u, v, ruv, guv, buv);

        v_uint8 r[4], g[4], b[4];

        for (int k = 0; k < 4; k++) {
            yRGBuvToRGB(vy[k], ruv, guv, buv, r[k], g[k], b[k]);
        }

        for (int k = 0; k < 4; k++)
            std::swap(r[k], b[k]);

        // [r0...], [r1...] => [r0, r1, r0, r1...], [r0, r1, r0, r1...]
        v_uint8 r0_0, r0_1, r1_0, r1_1;
        v_zip(r[0], r[1], r0_0, r0_1);
        v_zip(r[2], r[3], r1_0, r1_1);
        v_uint8 g0_0, g0_1, g1_0, g1_1;
        v_zip(g[0], g[1], g0_0, g0_1);
        v_zip(g[2], g[3], g1_0, g1_1);
        v_uint8 b0_0, b0_1, b1_0, b1_1;
        v_zip(b[0], b[1], b0_0, b0_1);
        v_zip(b[2], b[3], b1_0, b1_1);

        v_store_interleave(dstRGBx[0] + i * 3, b0_0, g0_0, r0_0);
        v_store_interleave(dstRGBx[0] + i * 3 + 3 * nlanes, b0_1, g0_1, r0_1);

        v_store_interleave(dstRGBx[1] + i * 3, b1_0, g1_0, r1_0);
        v_store_interleave(dstRGBx[1] + i * 3 + 3 * nlanes, b1_1, g1_1, r1_1);
    }
    //vx_cleanup();
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

                dstRGBx[y][3 * (i + x)] = r;
                dstRGBx[y][3 * (i + x) + 1] = g;
                dstRGBx[y][3 * (i + x) + 2] = b;
            }
        }
    }
}

template<typename isa_tag_t>
CV_ALWAYS_INLINE void i420ToRgbRowImpl(isa_tag_t, const uint8_t** srcY, const uint8_t* srcU,
                                       const uint8_t* srcV, uint8_t** dstRGBx, const int width) {
    int i = 0;

#if MANUAL_SIMD
    constexpr int nlanes = v_uint8::nlanes;

    for (; i <= width - 2 * nlanes; i += 2 * nlanes) {
        v_uint8 u = vx_load(srcU + i / 2);
        v_uint8 v = vx_load(srcV + i / 2);

        v_uint8 vy[4];
        v_load_deinterleave(srcY[0] + i, vy[0], vy[1]);
        v_load_deinterleave(srcY[1] + i, vy[2], vy[3]);

        v_int32 ruv[4], guv[4], buv[4];
        uvToRGBuv(u, v, ruv, guv, buv);

        v_uint8 r[4], g[4], b[4];

        for (int k = 0; k < 4; k++) {
            yRGBuvToRGB(vy[k], ruv, guv, buv, r[k], g[k], b[k]);
        }

        for (int k = 0; k < 4; k++)
            std::swap(r[k], b[k]);

        // [r0...], [r1...] => [r0, r1, r0, r1...], [r0, r1, r0, r1...]
        v_uint8 r0_0, r0_1, r1_0, r1_1;
        v_zip(r[0], r[1], r0_0, r0_1);
        v_zip(r[2], r[3], r1_0, r1_1);
        v_uint8 g0_0, g0_1, g1_0, g1_1;
        v_zip(g[0], g[1], g0_0, g0_1);
        v_zip(g[2], g[3], g1_0, g1_1);
        v_uint8 b0_0, b0_1, b1_0, b1_1;
        v_zip(b[0], b[1], b0_0, b0_1);
        v_zip(b[2], b[3], b1_0, b1_1);

        v_store_interleave(dstRGBx[0] + i * 3, b0_0, g0_0, r0_0);
        v_store_interleave(dstRGBx[0] + i * 3 + 3 * nlanes, b0_1, g0_1, r0_1);

        v_store_interleave(dstRGBx[1] + i * 3, b1_0, g1_0, r1_0);
        v_store_interleave(dstRGBx[1] + i * 3 + 3 * nlanes, b1_1, g1_1, r1_1);
    }
    //vx_cleanup();
#endif
    for (; i < width; i += 2) {
        uchar u = srcU[i / 2];
        uchar v = srcV[i / 2];
        int ruv, guv, buv;
        uvToRGBuv(u, v, ruv, guv, buv);

        for (int y = 0; y < 2; y++) {
            for (int x = 0; x < 2; x++) {
                uchar vy = srcY[y][i + x];
                uchar r, g, b;
                yRGBuvToRGB(vy, ruv, guv, buv, r, g, b);

                dstRGBx[y][3 * (i + x)] = r;
                dstRGBx[y][3 * (i + x) + 1] = g;
                dstRGBx[y][3 * (i + x) + 2] = b;
            }
        }
    }
}

//------------------------------------------------------------------------------

// vertical pass
template<typename T, typename A, typename I, typename W>
CV_ALWAYS_INLINE void downy(const T *src[], int inWidth, const MapperUnit<A, I>& ymap,
                            A yalpha, W vbuf[]) {
    int y_1st = ymap.index0;
    int ylast = ymap.index1 - 1;

    // yratio > 1, so at least 2 rows
    GAPI_DbgAssert(y_1st < ylast);

#if MANUAL_SIMD
    constexpr int nlanes = v_uint16::nlanes;
#endif

    // 1st and last rows
    {
        int w = 0;

#if MANUAL_SIMD
        if (std::is_same<T, uint8_t>::value) {
            for (; w <= inWidth - nlanes; w += nlanes) {
                v_uint16 vsrc0 = vx_load_expand(reinterpret_cast<const uint8_t*>(& src[0][w]));
                v_uint16 vsrc1 = vx_load_expand(reinterpret_cast<const uint8_t*>(& src[ylast - y_1st][w]));
                v_uint16 vres = v_mulhi(vsrc0 << 8, static_cast<Q0_16>(ymap.alpha0)) +
                                v_mulhi(vsrc1 << 8, static_cast<Q0_16>(ymap.alpha1));
                vx_store(reinterpret_cast<Q8_8*>(& vbuf[w]), vres);
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

#if MANUAL_SIMD
        if (std::is_same<T, uint8_t>::value) {
            for (; w <= inWidth - nlanes; w += nlanes) {
                v_uint16 vsrc = vx_load_expand(reinterpret_cast<const uint8_t*>(& src[i][w]));
                v_uint16 vres = vx_load(reinterpret_cast<Q8_8*>(& vbuf[w]));
                vres = vres + v_mulhi(vsrc << 8, static_cast<Q0_16>(yalpha));
                vx_store(reinterpret_cast<Q8_8*>(& vbuf[w]), vres);
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
CV_ALWAYS_INLINE void downx(T dst[], int outWidth, int xmaxdf, const I xindex[],
                            const A xalpha[], const W vbuf[]) {
// TO DO: try lambda here
#define HSUM(xmaxdf)                                 \
    for (int x = 0; x < outWidth; x++) {             \
        int      index =  xindex[x];                 \
        const A *alpha = &xalpha[x * xmaxdf];        \
                                                     \
        W sum = 0;                                   \
        for (int i = 0; i < xmaxdf; i++) {           \
            sum += mulaw(alpha[i], vbuf[index + i]); \
        }                                            \
                                                     \
        dst[x] = convert_cast<T>(sum);               \
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

template<typename isa_tag_t, typename T, typename A, typename I, typename W>
CV_ALWAYS_INLINE void calcRowAreaImpl(isa_tag_t, T dst[], const T *src[],
                                      const Size& inSz, const Size& outSz, A yalpha,
                                      const MapperUnit<A, I>& ymap, int xmaxdf,
                                      const I xindex[], const A xalpha[], W vbuf[]) {
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

//------------------------------------------------------------------------------

template <typename VecT, typename T>
CV_ALWAYS_INLINE void copyRow_Impl(const T in[], T out[], int length) {
    int l = 0;

#if MANUAL_SIMD
    const int nlanes = VecT::nlanes;

    auto copy_row = [](const T in[], T out[], int l) {
        VecT r = vx_load(&in[l]);
        vx_store(&out[l], r);
    };

    for (; l <= length - nlanes; l += nlanes) {
        copy_row(in, out, l);
    }

    if (l < length && length >= nlanes) {
        copy_row(in, out, length - nlanes);
        l = length;
    }
#endif

    for (; l < length; l++) {
        out[l] = in[l];
    }
}

// Resize (bi-linear, 32FC1)
template<typename isa_tag_t>
CV_ALWAYS_INLINE void calcRowLinear32FC1Impl(isa_tag_t,
                                             float *dst[],
                                             const float *src0[],
                                             const float *src1[],
                                             const float  alpha[],
                                             const int    mapsx[],
                                             const float  beta[],
                                             const Size& inSz,
                                             const Size& outSz,
                                             const int   lpi,
                                             const int) {
    bool xRatioEq1 = inSz.width == outSz.width;
    bool yRatioEq1 = inSz.height == outSz.height;

#if MANUAL_SIMD
    constexpr int nlanes = v_float32::nlanes;
#endif

    if (!xRatioEq1 && !yRatioEq1) {
        for (int line = 0; line < lpi; ++line) {
            float beta0 = beta[line];
            float beta1 = 1 - beta0;

            int x = 0;

#if MANUAL_SIMD
            v_float32 low1, high1, s00, s01;
            v_float32 low2, high2, s10, s11;
            for (; x <= outSz.width - nlanes; x += nlanes) {
                v_float32 alpha0 = vx_load(&alpha[x]);
                //  v_float32 alpha1 = 1.f - alpha0;

                v_gather_pairs(src0[line], mapsx, x, low1, high1);
                v_deinterleave(low1, high1, s00, s01);

                //  v_float32 res0 = s00*alpha0 + s01*alpha1;
                v_float32 res0 = v_fma(s00 - s01, alpha0, s01);

                v_gather_pairs(src1[line], mapsx, x, low2, high2);
                v_deinterleave(low2, high2, s10, s11);

                //  v_float32 res1 = s10*alpha0 + s11*alpha1;
                v_float32 res1 = v_fma(s10 - s11, alpha0, s11);

                //  v_float32 d = res0*beta0 + res1*beta1;
                v_float32 d = v_fma(res0 - res1, beta0, res1);

                vx_store(&dst[line][x], d);
            }
#endif

            for (; x < outSz.width; ++x) {
                float alpha0 = alpha[x];
                float alpha1 = 1 - alpha0;
                int   sx0 = mapsx[x];
                int   sx1 = sx0 + 1;
                float res0 = src0[line][sx0] * alpha0 + src0[line][sx1] * alpha1;
                float res1 = src1[line][sx0] * alpha0 + src1[line][sx1] * alpha1;
                dst[line][x] = beta0 * res0 + beta1 * res1;
            }
        }

    } else if (!xRatioEq1) {
        GAPI_DbgAssert(yRatioEq1);

        for (int line = 0; line < lpi; ++line) {
            int x = 0;

#if MANUAL_SIMD
            v_float32 low, high, s00, s01;
            for (; x <= outSz.width - nlanes; x += nlanes) {
                v_float32 alpha0 = vx_load(&alpha[x]);
                //  v_float32 alpha1 = 1.f - alpha0;

                v_gather_pairs(src0[line], mapsx, x, low, high);
                v_deinterleave(low, high, s00, s01);

                //  v_float32 d = s00*alpha0 + s01*alpha1;
                v_float32 d = v_fma(s00 - s01, alpha0, s01);

                vx_store(&dst[line][x], d);
            }
#endif

            for (; x < outSz.width; ++x) {
                float alpha0 = alpha[x];
                float alpha1 = 1 - alpha0;
                int   sx0 = mapsx[x];
                int   sx1 = sx0 + 1;
                dst[line][x] = src0[line][sx0] * alpha0 + src0[line][sx1] * alpha1;
            }
        }

    } else if (!yRatioEq1) {
        GAPI_DbgAssert(xRatioEq1);
        int length = inSz.width;  // == outSz.width

        for (int line = 0; line < lpi; ++line) {
            float beta0 = beta[line];
            float beta1 = 1 - beta0;

            int x = 0;

#if MANUAL_SIMD
            for (; x <= length - nlanes; x += nlanes) {
                v_float32 s0 = vx_load(&src0[line][x]);
                v_float32 s1 = vx_load(&src1[line][x]);

                //  v_float32 d = s0*beta0 + s1*beta1;
                v_float32 d = v_fma(s0 - s1, beta0, s1);

                vx_store(&dst[line][x], d);
            }
#endif

            for (; x < length; ++x) {
                dst[line][x] = beta0 * src0[line][x] + beta1 * src1[line][x];
            }
        }

    } else {
        GAPI_DbgAssert(xRatioEq1 && yRatioEq1);
        int length = inSz.width;  // == outSz.width
        for (int line = 0; line < lpi; ++line) {
            memcpy(dst[line], src0[line], length * sizeof(float));
        }
    }
}

template<typename isa_tag_t, typename scalar_t>
struct vector_type_of;

template<typename isa_tag_t, typename scalar_t>
using vector_type_of_t = typename vector_type_of<isa_tag_t, scalar_t>::type;

template<typename isa_tag_t> struct vector_type_of<isa_tag_t, uint8_t> { using type = v_uint8;  };
template<typename isa_tag_t> struct vector_type_of<isa_tag_t, float>   { using type = v_float32;};

template<typename isa_tag_t, typename T>
CV_ALWAYS_INLINE void chanToPlaneRowImpl(isa_tag_t, const T* in, const int chan,
                                         const int chs, T* out, const int length) {
    if (chs == 1) {
        copyRow_Impl<vector_type_of_t<isa_tag_t, T>, T>(in, out, length);
        return;
    }

    for (int x = 0; x < length; x++) {
        out[x] = in[x*chs + chan];
    }
}

template<typename isa_tag_t, typename T, int chs>
CV_ALWAYS_INLINE void splitRowImpl(isa_tag_t, const T* in, std::array<T*, chs>& outs, const int length) {
    static_assert(chs > 1 && chs < 5, "This number of channels isn't supported.");

    if (chs == 2) {
        splitRowC2_Impl<vector_type_of_t<isa_tag_t, T>, T>(in, outs[0], outs[1], length);
        return;
    } else if (chs == 3) {
        splitRowC3_Impl<vector_type_of_t<isa_tag_t, T>, T>(in, outs[0], outs[1], outs[2], length);
        return;
    } else {
        splitRowC4_Impl<vector_type_of_t<isa_tag_t, T>, T>(in, outs[0], outs[1], outs[2], outs[3], length);
        return;
    }
}

template<typename isa_tag_t, typename T, int chs>
CV_ALWAYS_INLINE void mergeRowImpl(isa_tag_t, const std::array<const T*, chs>& ins, T* out, const int length) {
    static_assert(chs > 1 && chs < 5, "This number of channels isn't supported.");

    if (chs == 2) {
        mergeRowC2_Impl<vector_type_of_t<isa_tag_t, T>, T>(ins[0], ins[1], out, length);
        return;
    } else if (chs == 3) {
        mergeRowC3_Impl<vector_type_of_t<isa_tag_t, T>, T>(ins[0], ins[1], ins[2], out, length);
        return;
    } else {
        mergeRowC4_Impl<vector_type_of_t<isa_tag_t, T>, T>(ins[0], ins[1], ins[2], ins[3], out, length);
        return;
    }
}
}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine

#endif  // IE_PREPROCESS_GAPI_KERNELS_SIMD_IMPL_H
