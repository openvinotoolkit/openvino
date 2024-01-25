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
#include "attn_quant_kernel.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

template <typename T, typename T2>
static void attn_quant_kernel(const ov::intel_cpu::PlainTensor& k_input,
                              const ov::intel_cpu::PlainTensor& v_input,
                              const ov::intel_cpu::PlainTensor& past_k_output,
                              const ov::intel_cpu::PlainTensor& past_v_output,
                              const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                              const ov::intel_cpu::PlainTensor& past_v_scale_zp) {
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3];
    parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t m) {
        auto p_k = &past_k_scale_zp.at<float>(b, h, m, false);
        auto p_v = &past_v_scale_zp.at<float>(b, h, m, false);
        quant_u8(&past_k_output.at<T2>(b, h, m, false),
                 &k_input.at<T>(b, h, m, false),
                 S,
                 p_k[0],
                 p_k[1]);
        quant_u8(&past_v_output.at<T2>(b, h, m, false),
                 &v_input.at<T>(b, h, m, false),
                 S,
                 p_v[0],
                 p_v[1]);
    });
}

void attn_quant(const ov::intel_cpu::PlainTensor& k_input,
                const ov::intel_cpu::PlainTensor& v_input,
                const ov::intel_cpu::PlainTensor& past_k_output,
                const ov::intel_cpu::PlainTensor& past_v_output,
                const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                const ov::intel_cpu::PlainTensor& past_v_scale_zp) {
    if (k_input.get_precision() == ov::element::f32 && past_k_output.get_precision() == ov::element::u8) {
        attn_quant_kernel<float, uint8_t>(k_input, v_input, past_k_output, past_v_output, past_k_scale_zp, past_v_scale_zp);
    } else if (k_input.get_precision() == ov::element::bf16 && past_k_output.get_precision() == ov::element::u8) {
        attn_quant_kernel<ov::bfloat16, uint8_t>(k_input, v_input, past_k_output, past_v_output, past_k_scale_zp, past_v_scale_zp);
    } else {
        OPENVINO_THROW("unsupport src type: ", k_input.get_precision(), ", dst type: ", past_k_output.get_precision(), " in attn_quant");
    }
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov