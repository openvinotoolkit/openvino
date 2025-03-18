// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlp_utils.hpp"

#include <cstring>
#if defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif
#include "../scaled_attn/transpose_kernel.hpp"

namespace ov::Extensions::Cpu::XARCH {

void llm_mlp_transpose_epi32_16x16(void* dst, void* src, int stride) {
    transpose_16x16_kernel(reinterpret_cast<uint32_t*>(dst),
                           reinterpret_cast<uint32_t*>(src),
                           16,
                           stride / sizeof(uint32_t));
}

template <typename T>
void llm_mlp_quantize_to_i8(T* psrc,
                            int src_stride,
                            int8_t* pdst,
                            int dst_stride,
                            int rows,
                            int cols,
                            float* p_scales,
                            float* p_zp,
                            bool asym) {
    auto clamp_i8 = [](float x) {
        auto v = static_cast<int>(std::round(x));
        if (v < -128) {
            return -128;
        }
        if (v > 127) {
            return 127;
        }
        return v;
    };

    for (int y = 0; y < rows; y++, psrc += src_stride, pdst += dst_stride) {
        int x = 0;
        float f_min, f_max;
#if defined(HAVE_AVX512F)
        auto v_max = mm512_uni_loadu_ps(psrc + 0);
        auto v_min = mm512_uni_loadu_ps(psrc + 0);
        for (; x + 16 <= cols; x += 16) {
            auto v = mm512_uni_loadu_ps(psrc + x);
            v_max = _mm512_max_ps(v, v_max);
            v_min = _mm512_min_ps(v, v_min);
        }
        f_max = _mm512_reduce_max_ps(v_max);
        f_min = _mm512_reduce_min_ps(v_min);
#else
        f_min = psrc[0];
        f_max = psrc[0];
#endif
        for (; x < cols; x++) {
            auto f_cur = static_cast<float>(psrc[x]);
            f_min = std::min(f_min, f_cur);
            f_max = std::max(f_max, f_cur);
        }
        // (q - z) * s = f
        //  (-128 - z) * s = f_min;
        //  ( 127 - z) * s = f_max;
        float scale, zp;
        if (f_max == f_min || std::isnan(f_max) || std::isnan(f_min)) {
            // special case
            p_zp[y] = 0;
            p_scales[y] = std::isnan(f_min) ? 0 : f_min;
#if defined(HAVE_AVX512F)
            auto vi8x16 = _mm_set1_epi8(1);
            for (; x + 16 <= cols; x += 16) {
                _mm_storeu_si128(reinterpret_cast<__m128i*>(pdst + x), vi8x16);
            }
#endif
            for (; x < cols; x++) {
                pdst[x] = 1;
            }
            continue;
        }
        if (asym) {
            scale = (f_max - f_min) / 255.0f;
            zp = 127 - (f_max / scale);
        } else {
            auto fx = std::max(std::abs(f_max), std::abs(f_min));
            scale = fx / 127.0f;
            zp = 0;
        }
        p_zp[y] = zp;
        p_scales[y] = scale;
        x = 0;
#if defined(HAVE_AVX512F)
        auto vscale = _mm512_set1_ps(1.0f / scale);
        auto vzp = _mm512_set1_ps(zp);
        for (; x + 16 <= cols; x += 16) {
            auto v = mm512_uni_loadu_ps(psrc + x);
            v = _mm512_fmadd_ps(v, vscale, vzp);
            auto vi32x16 = _mm512_cvtps_epi32(v);
            auto vi8x16 = _mm512_cvtepi32_epi8(vi32x16);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(pdst + x), vi8x16);
        }
#endif
        for (; x < cols; x++) {
            pdst[x] = clamp_i8(psrc[x] / scale + zp);
        }
    }
}

void llm_mlp_quantize_bf16_i8(ov::bfloat16* psrc,
                              int src_stride,
                              int8_t* pdst,
                              int dst_stride,
                              int rows,
                              int cols,
                              float* p_scales,
                              float* p_zp,
                              bool asym) {
    llm_mlp_quantize_to_i8(psrc, src_stride, pdst, dst_stride, rows, cols, p_scales, p_zp, asym);
}

void llm_mlp_quantize_f16_i8(ov::float16* psrc,
                             int src_stride,
                             int8_t* pdst,
                             int dst_stride,
                             int rows,
                             int cols,
                             float* p_scales,
                             float* p_zp,
                             bool asym) {
    llm_mlp_quantize_to_i8(psrc, src_stride, pdst, dst_stride, rows, cols, p_scales, p_zp, asym);
}

void llm_mlp_dequantize_i32_f32(int Batch,
                                int OC,
                                int32_t* src,
                                int stride_src,
                                float* dst,
                                int stride_dst,
                                float* p_src_scale_per_row,
                                float* p_src_zp_per_row,
                                float* p_wsum_per_oc,
                                float* p_wscale_per_oc,
                                bool asym) {
    for (int b = 0; b < Batch; b++, src += stride_src, dst += stride_dst) {
        float s1 = p_src_scale_per_row[b];
        float z1s1 = p_src_zp_per_row[b] * s1;
        int oc = 0;
#if defined(HAVE_AVX512F)
        if (asym) {
            auto vs1 = _mm512_set1_ps(s1);
            auto vz1s1 = _mm512_set1_ps(z1s1);
            for (; oc + 16 <= OC; oc += 16) {
                auto vs2 = mm512_uni_loadu_ps(p_wscale_per_oc + oc);
                auto vi32x16 = _mm512_loadu_si512(src + oc);
                auto vsrc = _mm512_cvtepi32_ps(vi32x16);
                auto vwsum = mm512_uni_loadu_ps(p_wsum_per_oc + oc);
                auto vwsum_z1s1 = _mm512_mul_ps(vwsum, vz1s1);
                vsrc = _mm512_fmsub_ps(vsrc, vs1, vwsum_z1s1);
                vsrc = _mm512_mul_ps(vsrc, vs2);
                mm512_uni_storeu_ps(dst + oc, vsrc);
            }
        } else {
            auto vs1 = _mm512_set1_ps(s1);
            for (; oc + 16 <= OC; oc += 16) {
                auto vs2 = mm512_uni_loadu_ps(p_wscale_per_oc + oc);
                auto vi32x16 = _mm512_loadu_si512(src + oc);
                auto vsrc = _mm512_cvtepi32_ps(vi32x16);
                vsrc = _mm512_mul_ps(vsrc, vs1);
                vsrc = _mm512_mul_ps(vsrc, vs2);
                mm512_uni_storeu_ps(dst + oc, vsrc);
            }
        }
#endif
        for (; oc < OC; oc++) {
            //
            // fdst = sum{(qi-z1)*s1 * wi*s2}
            //      = sum{[qi*wi] *(s1*s2) - z1*s1*wi*s2}
            //      = sum{qi*wi} *(s1*s2) - sum{wi}*(z1*s1*s2)
            //
            float s2 = p_wscale_per_oc[oc];
            if (asym) {
                dst[oc] = src[oc] * (s1 * s2) - p_wsum_per_oc[oc] * (z1s1 * s2);
            } else {
                dst[oc] = src[oc] * (s1 * s2);  // - sum_w *(z1 * s1 * s2);
            }
        }
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
