// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_gapi_kernels_impl.hpp"

namespace InferenceEngine {
namespace gapi {
namespace kernels {

//----------------------------------------------------------------------

typedef MapperUnit<float,   int> MapperUnit32F;
typedef MapperUnit<Q0_16, short> MapperUnit8U;

void calcRowArea_8U(uchar dst[], const uchar *src[], const Size &inSz, const Size &outSz,
    Q0_16 yalpha, const MapperUnit8U& ymap, int xmaxdf, const short xindex[], const Q0_16 xalpha[],
    Q8_8 vbuf[]);

void calcRowArea_32F(float dst[], const float *src[], const Size &inSz, const Size &outSz,
    float yalpha, const MapperUnit32F& ymap, int xmaxdf, const int xindex[], const float xalpha[],
    float vbuf[]);

#if USE_CVKL
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
                                     uint16_t vert_sum[]);
#endif

//----------------------------------------------------------------------

// Resize (bi-linear, 8U)
void calcRowLinear_8U(uint8_t *dst[],
                const uint8_t *src0[],
                const uint8_t *src1[],
                const short    alpha[],
                const short    clone[],
                const short    mapsx[],
                const short    beta[],
                      uint8_t  tmp[],
                const Size   & inSz,
                const Size   & outSz,
                      int      lpi);

void calcRowLinear_8UC3(std::array<std::array<uint8_t*, 4>, 3> &dst,
                  const uint8_t *src0[],
                  const uint8_t *src1[],
                  const short    alpha[],
                  const short    clone[],
                  const short    mapsx[],
                  const short    beta[],
                        uint8_t  tmp[],
                  const Size    &inSz,
                  const Size    &outSz,
                        int      lpi);

// Resize (bi-linear, 32F)
void calcRowLinear_32F(float *dst[],
                 const float *src0[],
                 const float *src1[],
                 const float  alpha[],
                 const int    mapsx[],
                 const float  beta[],
                 const Size & inSz,
                 const Size & outSz,
                       int    lpi);

//----------------------------------------------------------------------

void mergeRow_8UC2(const uint8_t in0[],
                   const uint8_t in1[],
                         uint8_t out[],
                             int length);

void mergeRow_8UC3(const uint8_t in0[],
                   const uint8_t in1[],
                   const uint8_t in2[],
                         uint8_t out[],
                             int length);

void mergeRow_8UC4(const uint8_t in0[],
                   const uint8_t in1[],
                   const uint8_t in2[],
                   const uint8_t in3[],
                         uint8_t out[],
                             int length);

void mergeRow_32FC2(const float in0[],
                    const float in1[],
                          float out[],
                            int length);

void mergeRow_32FC3(const float in0[],
                    const float in1[],
                    const float in2[],
                          float out[],
                            int length);

void mergeRow_32FC4(const float in0[],
                    const float in1[],
                    const float in2[],
                    const float in3[],
                          float out[],
                            int length);

void splitRow_8UC2(const uint8_t in[],
                         uint8_t out0[],
                         uint8_t out1[],
                             int length);

void splitRow_8UC3(const uint8_t in[],
                         uint8_t out0[],
                         uint8_t out1[],
                         uint8_t out2[],
                             int length);

void splitRow_8UC4(const uint8_t in[],
                         uint8_t out0[],
                         uint8_t out1[],
                         uint8_t out2[],
                         uint8_t out3[],
                             int length);

void splitRow_32FC2(const float in[],
                          float out0[],
                          float out1[],
                            int length);

void splitRow_32FC3(const float in[],
                          float out0[],
                          float out1[],
                          float out2[],
                            int length);

void splitRow_32FC4(const float in[],
                          float out0[],
                          float out1[],
                          float out2[],
                          float out3[],
                            int length);

void calculate_nv12_to_rgb(const  uchar **srcY,
                           const  uchar *srcUV,
                                  uchar **dstRGBx,
                                    int width);

void copyRow_8U(const uint8_t in[],
                uint8_t out[],
                int length);

void copyRow_32F(const float in[],
                 float out[],
                 int length);

}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine
