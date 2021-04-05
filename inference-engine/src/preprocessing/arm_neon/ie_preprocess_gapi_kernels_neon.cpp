// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_gapi_kernels_impl.hpp"
#include "ie_preprocess_gapi_kernels_neon.hpp"

#include <arm_neon.h>

#ifdef CV_NEON
#undef CV_NEON
#endif

#define CV_NEON 1

#ifdef CV_SIMD128
#undef CV_SIMD128
#endif

#define CV_SIMD128 1

#include "opencv_hal_intrin.hpp"
#include "ie_preprocess_gapi_kernels_simd_impl.hpp"

using namespace cv;

namespace InferenceEngine {
namespace gapi {
namespace kernels {
namespace neon {

void mergeRow_8UC2(const uint8_t in0[], const uint8_t in1[],
                   uint8_t out[], int length) {
    mergeRow_8UC2_Impl(in0, in1, out, length);
}

void mergeRow_8UC3(const uint8_t in0[], const uint8_t in1[],
                   const uint8_t in2[], uint8_t out[], int length) {
    mergeRow_8UC3_Impl(in0, in1, in2, out, length);
}

void mergeRow_8UC4(const uint8_t in0[], const uint8_t in1[], const uint8_t in2[],
                   const uint8_t in3[], uint8_t out[], int length) {
    mergeRow_8UC4_Impl(in0, in1, in2, in3, out, length);
}

void mergeRow_32FC2(const float in0[], const float in1[],
                    float out[], int length) {
    mergeRow_32FC2_Impl(in0, in1, out, length);
}

void mergeRow_32FC3(const float in0[], const float in1[], const float in2[],
                    float out[], int length) {
    mergeRow_32FC3_Impl(in0, in1, in2, out, length);
}

void mergeRow_32FC4(const float in0[], const float in1[],
                    const float in2[], const float in3[],
                    float out[], int length) {
    mergeRow_32FC4_Impl(in0, in1, in2, in3, out, length);
}

void splitRow_8UC2(const uint8_t in[], uint8_t out0[],
                   uint8_t out1[], int length) {
    splitRow_8UC2_Impl(in, out0, out1, length);
}

void splitRow_8UC3(const uint8_t in[], uint8_t out0[],
                   uint8_t out1[], uint8_t out2[], int length) {
    splitRow_8UC3_Impl(in, out0, out1, out2, length);
}

void splitRow_8UC4(const uint8_t in[], uint8_t out0[], uint8_t out1[],
                   uint8_t out2[], uint8_t out3[], int length) {
    splitRow_8UC4_Impl(in, out0, out1, out2, out3, length);
}

void splitRow_32FC2(const float in[], float out0[], float out1[], int length) {
    splitRow_32FC2_Impl(in, out0, out1, length);
}

void splitRow_32FC3(const float in[], float out0[], float out1[],
                    float out2[], int length) {
    splitRow_32FC3_Impl(in, out0, out1, out2, length);
}

void splitRow_32FC4(const float in[], float out0[], float out1[],
                    float out2[], float out3[], int length) {
    splitRow_32FC4_Impl(in, out0, out1, out2, out3, length);
}

void calculate_nv12_to_rgb(const  uchar **srcY,
                           const  uchar *srcUV,
                                  uchar **dstRGBx,
                                    int width) {
    calculate_nv12_to_rgb_impl(srcY, srcUV, dstRGBx, width);
}

void calculate_i420_to_rgb(const  uchar **srcY,
                           const  uchar *srcU,
                           const  uchar *srcV,
                                  uchar **dstRGBx,
                                    int width) {
    calculate_i420_to_rgb_impl(srcY, srcU, srcV, dstRGBx, width);
}

void calcRowArea_8U(uchar dst[], const uchar *src[], const Size& inSz,
                    const Size& outSz, Q0_16 yalpha, const MapperUnit8U &ymap,
                    int xmaxdf, const short xindex[], const Q0_16 xalpha[],
                    Q8_8 vbuf[]) {
    calcRowArea_impl(dst, src, inSz, outSz, yalpha, ymap, xmaxdf, xindex, xalpha, vbuf);
}

void calcRowArea_32F(float dst[], const float *src[], const Size& inSz,
                     const Size& outSz, float yalpha, const MapperUnit32F& ymap,
                     int xmaxdf, const int xindex[], const float xalpha[],
                     float vbuf[]) {
    calcRowArea_impl(dst, src, inSz, outSz, yalpha, ymap, xmaxdf, xindex, xalpha, vbuf);
}

void copyRow_8U(const uint8_t in[], uint8_t out[], int length) {
    copyRow_8U_impl(in, out, length);
}

void copyRow_32F(const float in[], float out[], int length) {
    copyRow_32F_impl(in, out, length);
}

// Resize (bi-linear, 32F)
void calcRowLinear_32F(float* dst[],
                       const float* src0[],
                       const float* src1[],
                       const float  alpha[],
                       const int    mapsx[],
                       const float  beta[],
                       const Size& inSz,
                       const Size& outSz,
                       const int   lpi) {
    calcRowLinear_32FC1(dst, src0, src1, alpha, mapsx, beta, inSz, outSz, lpi);
}

template<int chanNum>
CV_ALWAYS_INLINE void channels2planes_store(std::array<std::array<uint8_t*, 4>, chanNum>& dst,
                                            const uchar* src, const int width,
                                            const int line) {
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    GAPI_Assert(width >= nlanes);

    v_uint8 chan;
    int x = 0;
    for (;;) {
        for (; x <= width - nlanes && x >= 0; x += nlanes) {
            for (int c = 0; c < chanNum; ++c) {
                v_gather_channel<chanNum>(chan, &src[chanNum * x], c);
                vx_store(&dst[c][line][x], chan);
            }
        }

        if (x < width) {
            x = width - nlanes;
            continue;
        }
        break;
    }
}

CV_ALWAYS_INLINE void vertical_anyLPI(const uchar* src0, const uchar* src1,
                                      uchar* tmp, const int inLength,
                                      const short beta) {
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    GAPI_Assert(inLength >= nlanes);

    const int half_nlanes = nlanes/2;
    int w = 0;
    for (;;) {
        for (; w <= inLength - nlanes; w += nlanes) {
            v_int16 s0 = v_reinterpret_as_s16(vx_load_expand(&src0[w]));
            v_int16 s1 = v_reinterpret_as_s16(vx_load_expand(&src1[w]));
            v_int16 s2 = v_reinterpret_as_s16(vx_load_expand(&src0[w + half_nlanes]));
            v_int16 s3 = v_reinterpret_as_s16(vx_load_expand(&src1[w + half_nlanes]));
            v_int16 res1 = v_mulhrs(s0 - s1, beta) + s1;
            v_int16 res2 = v_mulhrs(s2 - s3, beta) + s3;

            vx_store(tmp + w, v_pack_u(res1, res2));
        }

        if (w < inLength) {
            w = inLength - nlanes;
            continue;
        }
        break;
    }
}

template<int chanNum>
CV_ALWAYS_INLINE void horizontal_anyLPI(std::array<std::array<uint8_t*, 4>, chanNum>& dst,
                                        const uchar* src, const short mapsx[],
                                        const short alpha[], const int width,
                                        const int line) {
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    const int half_nlanes = nlanes/2;
    GAPI_Assert(width >= half_nlanes);

    v_int16 t0, t1;//, t2, t3;
    int x = 0;
    for (;;) {
        for (; x <= width - half_nlanes && x >= 0; x += half_nlanes) {
            v_int16 a0 = vx_load(&alpha[x]);
            for (int c = 0; c < chanNum; ++c) {
                v_gather_channel<chanNum>(t0, src, &mapsx[x], c, 0);
                v_gather_channel<chanNum>(t1, src, &mapsx[x], c, 1);
                //v_gather_channel<chanNum>(t2, src, &mapsx[x + half_nlanes], c, 0);
                //v_gather_channel<chanNum>(t3, src, &mapsx[x + half_nlanes], c, 1);
                v_int16 res1 = v_mulhrs(t0 - t1, a0) + t1;
                //v_int16 res2 = v_mulhrs(t2 - t3, a0) + t3;
                //vx_store(&dst[c][line][x], v_pack_u(res1, res2));
                v_pack_u_store(&dst[c][line][x], res1);
            }
        }

        if (x < width) {
            //x = width - nlanes;
            x = width - half_nlanes;
            continue;
        }
        break;
    }
}

template<int chanNum>
CV_ALWAYS_INLINE void horizontal_4LPI(std::array<std::array<uint8_t*, 4>, chanNum>& dst,
                                      const uchar* tmp, const short mapsx[],
                                      const short clone[], const int length) {
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    const int half_nlanes = nlanes / 2;
    GAPI_Assert(length >= half_nlanes);

    const int shift = static_cast<int>(half_nlanes / 4);

    uchar _mask_horizontal[nlanes] = { 0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15 };
    v_uint8 hmask = vx_load(_mask_horizontal);

    v_uint8 val_0, val_1, val_2, val_3;
    int x = 0;
    for (;;) {
        for (; x <= length - half_nlanes && x >= 0; x += half_nlanes) {
            v_int16 a10 = vx_load(&clone[4 * x]);
            v_int16 a32 = vx_load(&clone[4 * (x + 2)]);
            v_int16 a54 = vx_load(&clone[4 * (x + 4)]);
            v_int16 a76 = vx_load(&clone[4 * (x + 6)]);

            for (int c = 0; c < chanNum; ++c) {
                v_gather_channel(val_0, tmp, &mapsx[x], chanNum, c, 0);
                v_gather_channel(val_1, tmp, &mapsx[x], chanNum, c, shift);
                v_gather_channel(val_2, tmp, &mapsx[x], chanNum, c, shift * 2);
                v_gather_channel(val_3, tmp, &mapsx[x], chanNum, c, shift * 3);

                v_int16 val0_0 = v_reinterpret_as_s16(v_expand_low(val_0));
                v_int16 val0_1 = v_reinterpret_as_s16(v_expand_low(val_1));
                v_int16 val0_2 = v_reinterpret_as_s16(v_expand_low(val_2));
                v_int16 val0_3 = v_reinterpret_as_s16(v_expand_low(val_3));

                v_int16 val1_0 = v_reinterpret_as_s16(v_expand_high(val_0));
                v_int16 val1_1 = v_reinterpret_as_s16(v_expand_high(val_1));
                v_int16 val1_2 = v_reinterpret_as_s16(v_expand_high(val_2));
                v_int16 val1_3 = v_reinterpret_as_s16(v_expand_high(val_3));

                v_int16 t0 = v_mulhrs(v_sub_wrap(val0_0, val1_0), a10);
                v_int16 t1 = v_mulhrs(v_sub_wrap(val0_1, val1_1), a32);
                v_int16 t2 = v_mulhrs(v_sub_wrap(val0_2, val1_2), a54);
                v_int16 t3 = v_mulhrs(v_sub_wrap(val0_3, val1_3), a76);

                v_int16 r0 = v_add_wrap(val1_0, t0);
                v_int16 r1 = v_add_wrap(val1_1, t1);
                v_int16 r2 = v_add_wrap(val1_2, t2);
                v_int16 r3 = v_add_wrap(val1_3, t3);

                v_uint8 q0 = v_pack_u(r0, r1);
                v_uint8 q1 = v_pack_u(r2, r3);

                v_uint8 q2 = v_shuffle(q0, hmask);
                v_uint8 q3 = v_shuffle(q1, hmask);

                v_uint8 q4 = v_blend<0xCC /*0b11001100*/>(q2, v_shift_left<4>(q3));
                v_uint8 q5 = v_blend<0xCC /*0b11001100*/>(v_shift_right<4>(q2), q3);

                v_store_low(&dst[c][0][x], q4);
                v_store_high(&dst[c][1][x], q4);
                v_store_low(&dst[c][2][x], q5);
                v_store_high(&dst[c][3][x], q5);
            }
        }

        if (x < length) {
            x = length - half_nlanes;
            continue;
        }
        break;
    }
}

template<int chanNum>
CV_ALWAYS_INLINE void calcRowLinear_8UC_Impl_(std::array<std::array<uint8_t*, 4>, chanNum>& dst,
                                              const uint8_t* src0[],
                                              const uint8_t* src1[],
                                              const short    alpha[],
                                              const short    clone[],  // 4 clones of alpha
                                              const short    mapsx[],
                                              const short    beta[],
                                                  uint8_t    tmp[],
                                              const Size&    inSz,
                                              const Size&    outSz,
                                              const int      lpi) {
    static_assert(v_uint8::nlanes == 16,
                  "The wide of NEON vector is 128 bits, so one vector contains 16 uchars");
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    constexpr int half_nlanes = nlanes / 2;

    bool xRatioEq = inSz.width == outSz.width;
    bool yRatioEq = inSz.height == outSz.height;

    if (!xRatioEq && !yRatioEq) {
        if (4 == lpi) {
            // vertical pass
            int inLength = inSz.width * chanNum;
            GAPI_Assert(inLength >= half_nlanes);

            v_int16 b0 = vx_setall_s16(beta[0]);
            v_int16 b1 = vx_setall_s16(beta[1]);
            v_int16 b2 = vx_setall_s16(beta[2]);
            v_int16 b3 = vx_setall_s16(beta[3]);

            uchar _mask_vertical[nlanes] = { 0, 8, 4, 12, 1, 9, 5, 13,
                                            2, 10, 6, 14, 3, 11, 7, 15 };
            v_uint8 vmask = vx_load(_mask_vertical);

            int w = 0;
            for (;;) {
                for (; w <= inLength - half_nlanes && w >= 0; w += half_nlanes) {
                    v_int16 val0_0 = v_reinterpret_as_s16(vx_load_expand(&src0[0][w]));
                    v_int16 val0_1 = v_reinterpret_as_s16(vx_load_expand(&src0[1][w]));
                    v_int16 val0_2 = v_reinterpret_as_s16(vx_load_expand(&src0[2][w]));
                    v_int16 val0_3 = v_reinterpret_as_s16(vx_load_expand(&src0[3][w]));

                    v_int16 val1_0 = v_reinterpret_as_s16(vx_load_expand(&src1[0][w]));
                    v_int16 val1_1 = v_reinterpret_as_s16(vx_load_expand(&src1[1][w]));
                    v_int16 val1_2 = v_reinterpret_as_s16(vx_load_expand(&src1[2][w]));
                    v_int16 val1_3 = v_reinterpret_as_s16(vx_load_expand(&src1[3][w]));

                    v_int16 t0 = v_mulhrs(v_sub_wrap(val0_0, val1_0), b0);
                    v_int16 t1 = v_mulhrs(v_sub_wrap(val0_1, val1_1), b1);
                    v_int16 t2 = v_mulhrs(v_sub_wrap(val0_2, val1_2), b2);
                    v_int16 t3 = v_mulhrs(v_sub_wrap(val0_3, val1_3), b3);

                    v_int16 r0 = v_add_wrap(val1_0, t0);
                    v_int16 r1 = v_add_wrap(val1_1, t1);
                    v_int16 r2 = v_add_wrap(val1_2, t2);
                    v_int16 r3 = v_add_wrap(val1_3, t3);

                    v_uint8 q0 = v_pack_u(r0, r1);
                    v_uint8 q1 = v_pack_u(r2, r3);

                    v_uint8 q2 = v_blend<0xCC /*0b11001100*/>(q0, v_shift_left<4>(q1));
                    v_uint8 q3 = v_blend<0xCC /*0b11001100*/>(v_shift_right<4>(q0), q1);

                    v_uint8 q4 = v_shuffle(q2, vmask);
                    v_uint8 q5 = v_shuffle(q3, vmask);

                    vx_store(&tmp[4 * w + 0], q4);
                    vx_store(&tmp[4 * w + 2 * half_nlanes], q5);
                }

                if (w < inLength) {
                    w = inLength - half_nlanes;
                    continue;
                }
                break;
            }

            // horizontal pass
            horizontal_4LPI<chanNum>(dst, tmp, mapsx, clone, outSz.width);
        } else {  // if any lpi
              int inLength = inSz.width * chanNum;

              for (int l = 0; l < lpi; ++l) {
                  short beta0 = beta[l];
                  const uchar* s0 = src0[l];
                  const uchar* s1 = src1[l];

                  // vertical pass
                  vertical_anyLPI(s0, s1, tmp, inLength, beta0);

                  // horizontal pass
                  horizontal_anyLPI<chanNum>(dst, tmp, mapsx, alpha, outSz.width, l);
              }
          }
    } else if (!xRatioEq) {
        GAPI_DbgAssert(yRatioEq);

        if (4 == lpi) {
            int inLength = inSz.width * chanNum;

            // vertical pass
            GAPI_Assert(inLength >= nlanes);
            v_uint8 s0, s1, s2, s3;
            int w = 0;
            for (;;) {
                for (; w <= inLength - nlanes; w += nlanes) {
                    s0 = vx_load(&src0[0][w]);
                    s1 = vx_load(&src0[1][w]);
                    s2 = vx_load(&src0[2][w]);
                    s3 = vx_load(&src0[3][w]);
                    v_store_interleave(&tmp[lpi * w], s0, s1, s2, s3);
                }

                if (w < inLength) {
                    w = inLength - nlanes;
                    continue;
                }
                break;
            }

            // horizontal pass
            horizontal_4LPI<chanNum>(dst, tmp, mapsx, clone, outSz.width);
        } else {  // any LPI
            for (int l = 0; l < lpi; ++l) {
                const uchar* src = src0[l];

                // horizontal pass
                horizontal_anyLPI<chanNum>(dst, src, mapsx, alpha, outSz.width, l);
            }
        }
    } else if (!yRatioEq) {
        GAPI_DbgAssert(xRatioEq);
        int inLength = inSz.width*chanNum;  // == outSz.width

        for (int l = 0; l < lpi; ++l) {
            short beta0 = beta[l];
            const uchar* s0 = src0[l];
            const uchar* s1 = src1[l];

            // vertical pass
            vertical_anyLPI(s0, s1, tmp, inLength, beta0);

            //split channels to planes and store
            channels2planes_store<chanNum>(dst, tmp, outSz.width, l);
        }
    } else {
        GAPI_DbgAssert(xRatioEq && yRatioEq);

        //split channels to planes and store
        for (int l = 0; l < lpi; ++l) {
            const uchar* src = src0[l];
            channels2planes_store<chanNum>(dst, src, outSz.width, l);
        }
    }
}

// Resize (bi-linear, 8UC3)
void calcRowLinear_8U(C3, std::array<std::array<uint8_t*, 4>, 3>& dst,
                      const uint8_t* src0[],
                      const uint8_t* src1[],
                      const short    alpha[],
                      const short    clone[],  // 4 clones of alpha
                      const short    mapsx[],
                      const short    beta[],
                          uint8_t    tmp[],
                      const Size&    inSz,
                      const Size&    outSz,
                        const int    lpi) {
    constexpr int chanNum = 3;
    calcRowLinear_8UC_Impl_<chanNum>(dst, src0, src1, alpha, clone, mapsx,
                                     beta, tmp, inSz, outSz, lpi);
}

// Resize (bi-linear, 8UC4)
void calcRowLinear_8U(C4, std::array<std::array<uint8_t*, 4>, 4>& dst,
                      const uint8_t* src0[],
                      const uint8_t* src1[],
                      const short    alpha[],
                      const short    clone[],  // 4 clones of alpha
                      const short    mapsx[],
                      const short    beta[],
                          uint8_t    tmp[],
                      const Size&    inSz,
                      const Size&    outSz,
                      const int      lpi) {
    constexpr int chanNum = 4;
    calcRowLinear_8UC_Impl_<chanNum>(dst, src0, src1, alpha, clone, mapsx,
                                     beta, tmp, inSz, outSz, lpi);
}

CV_ALWAYS_INLINE void horizontal_4LPI(uint8_t* dst[],
                                      const uchar* tmp, const short mapsx[],
                                      const short clone[], const int length) {
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    const int half_nlanes = nlanes / 2;
    GAPI_Assert(length >= half_nlanes);

    uchar _mask_horizontal[nlanes] = { 0, 4, 8, 12, 2, 6, 10, 14,
                                      1, 5, 9, 13, 3, 7, 11, 15 };
    v_uint8 hmask = vx_load(_mask_horizontal);
    int x = 0;
    for (;;) {
        for (; x <= length - half_nlanes; x += half_nlanes) {
            v_int16 a10 = vx_load(&clone[4 * x]);
            v_int16 a32 = vx_load(&clone[4 * (x + 2)]);
            v_int16 a54 = vx_load(&clone[4 * (x + 4)]);
            v_int16 a76 = vx_load(&clone[4 * (x + 6)]);

            v_uint8 val_0 = v_gather_lines(tmp, &mapsx[x]);
            v_uint8 val_1 = v_gather_lines(tmp, &mapsx[x + 2]);
            v_uint8 val_2 = v_gather_lines(tmp, &mapsx[x + 4]);
            v_uint8 val_3 = v_gather_lines(tmp, &mapsx[x + 6]);

            v_int16 val0_0 = v_reinterpret_as_s16(v_expand_low(val_0));
            v_int16 val0_1 = v_reinterpret_as_s16(v_expand_low(val_1));
            v_int16 val0_2 = v_reinterpret_as_s16(v_expand_low(val_2));
            v_int16 val0_3 = v_reinterpret_as_s16(v_expand_low(val_3));

            v_int16 val1_0 = v_reinterpret_as_s16(v_expand_high(val_0));
            v_int16 val1_1 = v_reinterpret_as_s16(v_expand_high(val_1));
            v_int16 val1_2 = v_reinterpret_as_s16(v_expand_high(val_2));
            v_int16 val1_3 = v_reinterpret_as_s16(v_expand_high(val_3));

            v_int16 t0 = v_mulhrs(v_sub_wrap(val0_0, val1_0), a10);
            v_int16 t1 = v_mulhrs(v_sub_wrap(val0_1, val1_1), a32);
            v_int16 t2 = v_mulhrs(v_sub_wrap(val0_2, val1_2), a54);
            v_int16 t3 = v_mulhrs(v_sub_wrap(val0_3, val1_3), a76);

            v_int16 r0 = v_add_wrap(val1_0, t0);
            v_int16 r1 = v_add_wrap(val1_1, t1);
            v_int16 r2 = v_add_wrap(val1_2, t2);
            v_int16 r3 = v_add_wrap(val1_3, t3);

            v_uint8 q0 = v_pack_u(r0, r1);
            v_uint8 q1 = v_pack_u(r2, r3);

            v_uint8 q2 = v_shuffle(q0, hmask);
            v_uint8 q3 = v_shuffle(q1, hmask);

            v_uint8 q4 = v_blend<0xCC /*0b11001100*/>(q2, v_shift_left<4>(q3));
            v_uint8 q5 = v_blend<0xCC /*0b11001100*/>(v_shift_right<4>(q2), q3);

            v_store_low(&dst[0][x],  q4);
            v_store_high(&dst[1][x], q4);
            v_store_low(&dst[2][x],  q5);
            v_store_high(&dst[3][x], q5);
        }

        if (x < length) {
            x = length - half_nlanes;
            continue;
        }
        break;
    }
}

CV_ALWAYS_INLINE void horizontal_anyLPI(uint8_t* dst[],
                                        const uchar* src, const short mapsx[],
                                        const short alpha[], const int length,
                                        const int line) {
    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    const int half_nlanes = nlanes / 2;
    GAPI_Assert(length >= half_nlanes);
    v_int16 t0, t1;
    int x = 0;
    for (;;) {
        for (; x <= length - half_nlanes; x += half_nlanes) {
            v_int16 a0 = vx_load(&alpha[x]);
            v_uint8 t = v_gather_pairs(src, &mapsx[x]);

            v_deinterleave_expand(t, t0, t1);
            v_int16 d = v_mulhrs(t0 - t1, a0) + t1;
            v_pack_u_store(&dst[line][x], d);
        }

        if (x < length) {
            x = length - half_nlanes;
            continue;
        }
        break;
    }
}

// 8UC1 Resize (bi-linear)
void calcRowLinear_8UC1(uint8_t* dst[],
                        const uint8_t* src0[],
                        const uint8_t* src1[],
                        const short    alpha[],
                        const short    clone[],  // 4 clones of alpha
                        const short    mapsx[],
                        const short    beta[],
                            uint8_t    tmp[],
                        const Size&    inSz,
                        const Size&    outSz,
                        const int      lpi) {
    static_assert(v_uint8::nlanes == 16,
                  "The wide of NEON vector is 128 bits, so one vector contains 16 uchars");

    constexpr int nlanes = static_cast<int>(v_uint8::nlanes);
    constexpr int half_nlanes = nlanes / 2;

    bool xRatioEq = inSz.width == outSz.width;
    bool yRatioEq = inSz.height == outSz.height;

    if (!xRatioEq && !yRatioEq) {
        GAPI_Assert(inSz.width >= half_nlanes);

        if (4 == lpi) {
            // vertical pass
            v_int16 b0 = vx_setall_s16(beta[0]);
            v_int16 b1 = vx_setall_s16(beta[1]);
            v_int16 b2 = vx_setall_s16(beta[2]);
            v_int16 b3 = vx_setall_s16(beta[3]);

            uchar _mask_vertical[nlanes] = { 0, 8, 4, 12, 1, 9, 5, 13,
                                            2, 10, 6, 14, 3, 11, 7, 15 };
            v_uint8 vmask = vx_load(_mask_vertical);

            int w = 0;
            for (;;) {
                for (; w <= inSz.width - half_nlanes; w += half_nlanes) {
                    v_int16 val0_0 = v_reinterpret_as_s16(vx_load_expand(&src0[0][w]));
                    v_int16 val0_1 = v_reinterpret_as_s16(vx_load_expand(&src0[1][w]));
                    v_int16 val0_2 = v_reinterpret_as_s16(vx_load_expand(&src0[2][w]));
                    v_int16 val0_3 = v_reinterpret_as_s16(vx_load_expand(&src0[3][w]));

                    v_int16 val1_0 = v_reinterpret_as_s16(vx_load_expand(&src1[0][w]));
                    v_int16 val1_1 = v_reinterpret_as_s16(vx_load_expand(&src1[1][w]));
                    v_int16 val1_2 = v_reinterpret_as_s16(vx_load_expand(&src1[2][w]));
                    v_int16 val1_3 = v_reinterpret_as_s16(vx_load_expand(&src1[3][w]));

                    v_int16 t0 = v_mulhrs(v_sub_wrap(val0_0, val1_0), b0);
                    v_int16 t1 = v_mulhrs(v_sub_wrap(val0_1, val1_1), b1);
                    v_int16 t2 = v_mulhrs(v_sub_wrap(val0_2, val1_2), b2);
                    v_int16 t3 = v_mulhrs(v_sub_wrap(val0_3, val1_3), b3);

                    v_int16 r0 = v_add_wrap(val1_0, t0);
                    v_int16 r1 = v_add_wrap(val1_1, t1);
                    v_int16 r2 = v_add_wrap(val1_2, t2);
                    v_int16 r3 = v_add_wrap(val1_3, t3);

                    v_uint8 q0 = v_pack_u(r0, r1);
                    v_uint8 q1 = v_pack_u(r2, r3);

                    v_uint8 q2 = v_blend<0xCC /*0b11001100*/>(q0, v_shift_left<4>(q1));
                    v_uint8 q3 = v_blend<0xCC /*0b11001100*/>(v_shift_right<4>(q0), q1);

                    v_uint8 q4 = v_shuffle(q2, vmask);
                    v_uint8 q5 = v_shuffle(q3, vmask);

                    vx_store(&tmp[4 * w + 0], q4);
                    vx_store(&tmp[4 * w + 2 * half_nlanes], q5);
                }

                if (w < inSz.width) {
                    w = inSz.width - half_nlanes;
                    continue;
                }
                break;
            }

            // horizontal pass
             horizontal_4LPI(dst, tmp, mapsx, clone, outSz.width);
        } else {  // if any lpi
            for (int l = 0; l < lpi; ++l) {
                short beta0 = beta[l];
                const uchar* s0 = src0[l];
                const uchar* s1 = src1[l];

                // vertical pass
                vertical_anyLPI(s0, s1, tmp, inSz.width, beta0);

                // horizontal pass
                horizontal_anyLPI(dst, tmp, mapsx, alpha, outSz.width, l);
            }
        }  // if lpi == 4

    } else if (!xRatioEq) {
        GAPI_DbgAssert(yRatioEq);
        GAPI_Assert(inSz.width >= nlanes);

        if (4 == lpi) {
            // vertical pass
            int w = 0;
            for (;;) {
                for (; w <= inSz.width - nlanes; w += nlanes) {
                    v_uint8 s0 = vx_load(&src0[0][w]);
                    v_uint8 s1 = vx_load(&src0[1][w]);
                    v_uint8 s2 = vx_load(&src0[2][w]);
                    v_uint8 s3 = vx_load(&src0[3][w]);
                    v_store_interleave(&tmp[4 * w], s0, s1, s2, s3);
                }

                if (w < inSz.width) {
                    w = inSz.width - nlanes;
                    continue;
                }
                break;
            }

            // horizontal pass
            horizontal_4LPI(dst, tmp, mapsx, clone, outSz.width);
        } else {  // any LPI
            GAPI_Assert(outSz.width >= half_nlanes);
            for (int l = 0; l < lpi; ++l) {
                const uchar* src = src0[l];

                // horizontal pass
                horizontal_anyLPI(dst, src, mapsx, alpha, outSz.width, l);
            }
        }

    } else if (!yRatioEq) {
        GAPI_DbgAssert(xRatioEq);
        int length = inSz.width;  // == outSz.width

        for (int l = 0; l < lpi; ++l) {
            short beta0 = beta[l];
            const uchar* s0 = src0[l];
            const uchar* s1 = src1[l];

            // vertical pass
            vertical_anyLPI(s0, s1, dst[l], length, beta0);
        }

    } else {
        GAPI_DbgAssert(xRatioEq && yRatioEq);
        int length = inSz.width;  // == outSz.width

        for (int l = 0; l < lpi; ++l) {
            memcpy(dst[l], src0[l], length);
        }
    }
}
}  // namespace neon
}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine
