// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_preprocess_gapi_kernels.hpp"
#include "ie_preprocess_gapi_kernels_impl.hpp"
#include  <type_traits>

namespace InferenceEngine {
namespace gapi {
namespace kernels {
namespace neon {

using C3 = std::integral_constant<int, 3>;
using C4 = std::integral_constant<int, 4>;
//-----------------------------------------------------------------------------

typedef MapperUnit<float,   int> MapperUnit32F;
typedef MapperUnit<Q0_16, short> MapperUnit8U;

void calcRowArea_8U(uchar dst[], const uchar *src[], const Size &inSz, const Size &outSz,
                    Q0_16 yalpha, const MapperUnit8U& ymap, int xmaxdf, const short xindex[],
                    const Q0_16 xalpha[], Q8_8 vbuf[]);

void calcRowArea_32F(float dst[], const float *src[], const Size &inSz, const Size &outSz,
                     float yalpha, const MapperUnit32F& ymap, int xmaxdf, const int xindex[],
                     const float xalpha[], float vbuf[]);
}  // namespace neon

template<typename isa_tag_t, typename T>
void chanToPlaneRowImpl(isa_tag_t, const T* in, const int chan, const int chs, T* out, const int length);

extern template void chanToPlaneRowImpl(neon_tag, const uint8_t* in, const int chan, const int chs, uint8_t* out, const int length);
extern template void chanToPlaneRowImpl(neon_tag, const float*   in, const int chan, const int chs, float  * out, const int length);

template<typename isa_tag_t>
void nv12ToRgbRowImpl(isa_tag_t, const uint8_t** y_rows, const uint8_t* uv_row, uint8_t** out_rows, const int buf_width);

extern template void nv12ToRgbRowImpl(neon_tag, const uint8_t** y_rows, const uint8_t* uv_row, uint8_t** out_rows, const int buf_width);

template<typename isa_tag_t>
void i420ToRgbRowImpl(isa_tag_t, const uint8_t** y_rows, const uint8_t* u_row,
                      const uint8_t* v_row, uint8_t** out_rows, const int buf_width);

extern template void i420ToRgbRowImpl(neon_tag, const uint8_t** y_rows, const uint8_t* u_row,
                                      const uint8_t* v_row, uint8_t** out_rows, const int buf_width);

template<typename isa_tag_t, typename T, int chs>
void splitRowImpl(isa_tag_t, const T* in, std::array<T*, chs>& outs, const int length);

extern template void splitRowImpl<neon_tag, uint8_t, 2>(neon_tag, const uint8_t* in, std::array<uint8_t*, 2>& outs, const int length);
extern template void splitRowImpl<neon_tag, float, 2>(neon_tag, const float* in, std::array<float*, 2>& outs, const int length);
extern template void splitRowImpl<neon_tag, uint8_t, 3>(neon_tag, const uint8_t* in, std::array<uint8_t*, 3>& outs, const int length);
extern template void splitRowImpl<neon_tag, float, 3>(neon_tag, const float* in, std::array<float*, 3>& outs, const int length);
extern template void splitRowImpl<neon_tag, uint8_t, 4>(neon_tag, const uint8_t* in, std::array<uint8_t*, 4>& outs, const int length);
extern template void splitRowImpl<neon_tag, float, 4>(neon_tag, const float* in, std::array<float*, 4>& outs, const int length);

template<typename isa_tag_t, typename T, int chs>
void mergeRowImpl(isa_tag_t, const std::array<const T*, chs>& ins, T* out, const int length);

extern template void mergeRowImpl<neon_tag, uint8_t, 2>(neon_tag, const std::array<const uint8_t*, 2>& ins, uint8_t* out, const int length);
extern template void mergeRowImpl<neon_tag, float, 2>(neon_tag, const std::array<const float*, 2>& ins, float* out, const int length);
extern template void mergeRowImpl<neon_tag, uint8_t, 3>(neon_tag, const std::array<const uint8_t*, 3>& ins, uint8_t* out, const int length);
extern template void mergeRowImpl<neon_tag, float, 3>(neon_tag, const std::array<const float*, 3>& ins, float* out, const int length);
extern template void mergeRowImpl<neon_tag, uint8_t, 4>(neon_tag, const std::array<const uint8_t*, 4>& ins, uint8_t* out, const int length);
extern template void mergeRowImpl<neon_tag, float, 4>(neon_tag, const std::array<const float*, 4>& ins, float* out, const int length);

template<typename isa_tag_t>
bool calcRowLinear8UC1Impl(isa_tag_t, uint8_t* dst[], const uint8_t* src0[], const uint8_t* src1[],
                           const short alpha[], const short clone[], const short mapsx[],
                           const short beta[], uint8_t tmp[], const Size& inSz,
                           const Size& outSz, const int lpi, const int l);

template<typename isa_tag_t>
void calcRowLinear32FC1Impl(isa_tag_t, float* dst[], const float* src0[], const float* src1[],
                            const float alpha[], const int mapsx[],
                            const float beta[], const Size& inSz, const Size& outSz,
                            const int lpi, const int l);

extern template void calcRowLinear32FC1Impl(neon_tag, float* dst[], const float* src0[],
                                            const float* src1[], const float alpha[],
                                            const int mapsx[], const float beta[],
                                            const Size& inSz, const Size& outSz,
                                            const int lpi, const int l);

template<typename isa_tag_t, int chs>
bool calcRowLinear8UC3C4Impl(isa_tag_t, std::array<std::array<uint8_t*, 4>, chs>& dst,
                             const uint8_t* src0[], const uint8_t* src1[],
                             const short alpha[], const short clone[], const short mapsx[],
                             const short beta[], uint8_t tmp[], const Size& inSz,
                             const Size& outSz, const int lpi, const int l);
}  // namespace kernels
}  // namespace gapi
}  // namespace InferenceEngine
