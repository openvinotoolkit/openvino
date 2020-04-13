// Copyright (C) 2019-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <utility>
#include <cstring>

#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_gapi_kernels_impl.hpp"
#include "ie_preprocess_gapi_kernels_avx512.hpp"

#include <immintrin.h>

#ifdef CV_AVX512_SKX
#undef CV_AVX512_SKX
#endif

#define CV_AVX512_SKX 1

#define CV_CPU_HAS_SUPPORT_SSE2 1

#ifdef CV_SIMD512
#undef CV_SIMD512
#endif

#define CV_SIMD512 1

#include "opencv_hal_intrin.hpp"
#include "ie_preprocess_gapi_kernels_simd_impl.hpp"

using namespace cv;

namespace InferenceEngine {

namespace gapi {

namespace kernels {

namespace avx512 {
//----------------------------------------------------------------------

static inline v_uint16x32 v_expand_low(const v_uint8x64& a) {
    return v_uint16x32(_mm512_unpacklo_epi8(a.val, _mm512_setzero_si512()));
}

static inline v_uint16x32 v_expand_high(const v_uint8x64& a) {
    return v_uint16x32(_mm512_unpackhi_epi8(a.val, _mm512_setzero_si512()));
}

//------------------------------------------------------------------------------

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

}  // namespace avx512
}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine
