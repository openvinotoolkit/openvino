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
    for (; i < n; i++) {
        float tmp = b[i];
        max = std::max(max, tmp);
        min = std::min(min, tmp);
    }
    scale = (max - min) / 255;
    zp = -min / scale;
    for (size_t i = 0; i < n; i++) {
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