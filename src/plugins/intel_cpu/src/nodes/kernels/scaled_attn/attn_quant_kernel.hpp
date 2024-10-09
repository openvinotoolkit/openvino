#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#include <arm_neon.h>
#endif

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template<typename TDST>
void attn_dequant_u8_kernel(const uint8_t* src, TDST* dst, size_t n, float scale, float zp) {
    size_t i = 0;
    // loadu_si128/epi64 does not support const qualifier
    uint8_t* src_nc = const_cast<uint8_t*>(src);

#if defined(HAVE_AVX512F)
    auto v_zp = _mm512_set1_ps(zp);
    auto v_scale = _mm512_set1_ps(scale);
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_nc + i));
        auto v0_512 = _mm512_cvtepu8_epi32(v0_128);
        auto v0_value = _mm512_cvtepi32_ps(v0_512);
        v0_value = _mm512_sub_ps(v0_value, v_zp);
        auto v0_out = _mm512_mul_ps(v0_value, v_scale);
        mm512_uni_storeu_ps(dst + i, v0_out);
    }
#elif defined(HAVE_AVX2)
    auto v_zp = _mm256_set1_ps(zp);
    auto v_scale = _mm256_set1_ps(scale);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(src_nc + i));
        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        v0_value = _mm256_sub_ps(v0_value, v_zp);
        auto v0_out = _mm256_mul_ps(v0_value, v_scale);
        mm256_uni_storeu_ps(dst + i, v0_out);
    }
#elif defined(OPENVINO_ARCH_ARM64)
    float32x4_t v_zp = vdupq_n_f32(zp);
    float32x4_t v_scale = vdupq_n_f32(scale);
    for (; i + 16 <= n; i += 16) {
        uint8x16_t v0_u8 = vld1q_u8(src_nc + i);
        uint16x8_t v0_u16_low = vmovl_u8(vget_low_u8(v0_u8));
        uint16x8_t v0_u16_high = vmovl_u8(vget_high_u8(v0_u8));
        
        int32x4_t v0_i32_low1 = vmovl_u16(vget_low_u16(v0_u16_low));
        int32x4_t v0_i32_low2 = vmovl_u16(vget_high_u16(v0_u16_low));
        int32x4_t v0_i32_high1 = vmovl_u16(vget_low_u16(v0_u16_high));
        int32x4_t v0_i32_high2 = vmovl_u16(vget_high_u16(v0_u16_high));
        
        float32x4_t v0_ps_low1 = vcvtq_f32_s32(v0_i32_low1);
        float32x4_t v0_ps_low2 = vcvtq_f32_s32(v0_i32_low2);
        float32x4_t v0_ps_high1 = vcvtq_f32_s32(v0_i32_high1);
        float32x4_t v0_ps_high2 = vcvtq_f32_s32(v0_i32_high2);

        v0_ps_low1 = vmlsq_f32(v0_ps_low1, v_zp, v_scale);
        v0_ps_low2 = vmlsq_f32(v0_ps_low2, v_zp, v_scale);
        v0_ps_high1 = vmlsq_f32(v0_ps_high1, v_zp, v_scale);
        v0_ps_high2 = vmlsq_f32(v0_ps_high2, v_zp, v_scale);

        vst1q_f32(dst + i, v0_ps_low1);
        vst1q_f32(dst + i + 4, v0_ps_low2);
        vst1q_f32(dst + i + 8, v0_ps_high1);
        vst1q_f32(dst + i + 12, v0_ps_high2);
    }
#endif
    for (; i < n; ++i) {
        float tmp = src_nc[i];
        tmp = (tmp - zp) * scale;
        dst[i] = tmp;
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov