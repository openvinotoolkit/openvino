// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/parallel.hpp"
#include "common.hpp"
#include "attn_quant.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

template<typename TA, typename TB>
static void quant_u8(TA* a, TB* b, size_t n, float& zp, float& scale) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
#if defined(HAVE_AVX2)
    auto v_max = _mm256_set1_ps(-FLT_MAX);
    auto v_min = _mm256_set1_ps(FLT_MAX);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(b + i);
        v_max = _mm256_max_ps(v_max, v);
        v_min = _mm256_min_ps(v_min, v);
    }
    hmax(v_max);
    hmin(v_min);
    max = _mm256_cvtss_f32(v_max);
    min = _mm256_cvtss_f32(v_min);
#endif
    for (; i < n; i++) {
        float tmp = b[i];
        max = std::max(max, tmp);
        min = std::min(min, tmp);
    }
    scale = (max - min) / 255;
    zp = -min / scale;

#if defined(HAVE_AVX2)
    i = 0;
    auto v_scale = _mm256_set1_ps(1 / scale);
    auto v_zp = _mm256_set1_ps(zp);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(b + i);
        v = _mm256_mul_ps(v, v_scale);
        v = _mm256_add_ps(v, v_zp);
        v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
        auto v_i32 = _mm256_cvtps_epi32(v);

        auto high4 = _mm256_extractf128_si256(v_i32, 1);
        auto low4 = _mm256_castsi256_si128(v_i32);
        auto packed = _mm_packs_epi32(low4, high4);
        packed = _mm_packus_epi16(packed, packed);
        _mm_storeu_si64(a + i, packed);
    }
#endif
    for (; i < n; i++) {
        float tmp = b[i];
        a[i] = tmp / scale + zp;
    }
}

template <typename T, typename T2>
void attn_quant_kernel(const ov::intel_cpu::PlainTensor& k_input,
                       const ov::intel_cpu::PlainTensor& v_input,
                       const ov::intel_cpu::PlainTensor& past_k_output,
                       const ov::intel_cpu::PlainTensor& past_v_output,
                       const ov::intel_cpu::PlainTensor& past_k_zp,
                       const ov::intel_cpu::PlainTensor& past_v_zp,
                       const ov::intel_cpu::PlainTensor& past_k_scale,
                       const ov::intel_cpu::PlainTensor& past_v_scale) {
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3];
    parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t m) {
        quant_u8(&past_k_output.at<T2>({b, h, m, 0}),
                 &k_input.at<T>({b, h, m, 0}),
                 S,
                 past_k_zp.at<float>({b, h, m}),
                 past_k_scale.at<float>({b, h, m}));
        quant_u8(&past_v_output.at<T2>({b, h, m, 0}),
                 &v_input.at<T>({b, h, m, 0}),
                 S,
                 past_v_zp.at<float>({b, h, m}),
                 past_v_scale.at<float>({b, h, m}));
    });
}

void attn_quant(const ov::intel_cpu::PlainTensor& k_input,
                const ov::intel_cpu::PlainTensor& v_input,
                const ov::intel_cpu::PlainTensor& past_k_output,
                const ov::intel_cpu::PlainTensor& past_v_output,
                const ov::intel_cpu::PlainTensor& past_k_zp,
                const ov::intel_cpu::PlainTensor& past_v_zp,
                const ov::intel_cpu::PlainTensor& past_k_scale,
                const ov::intel_cpu::PlainTensor& past_v_scale) {
    if (k_input.get_precision() == ov::element::f32 && past_k_output.get_precision() == ov::element::u8) {
        attn_quant_kernel<float, uint8_t>(k_input, v_input, past_k_output, past_v_output, past_k_zp, past_v_zp, past_k_scale, past_v_scale);
    } else if (k_input.get_precision() == ov::element::bf16 && past_k_output.get_precision() == ov::element::u8) {
        attn_quant_kernel<ov::bfloat16, uint8_t>(k_input, v_input, past_k_output, past_v_output, past_k_zp, past_v_zp, past_k_scale, past_v_scale);
    } else {
        OPENVINO_THROW("unsupport src type: ", k_input.get_precision(), ", dst type: ", past_k_output.get_precision(), " in attn_quant");
    }
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov