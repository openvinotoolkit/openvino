// Copyright (C) 2018-2024 Intel Corporation
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
#include "attn_memcpy.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

// float16 <- float
template<typename TA, typename TB>
void attn_copy(TA* a, TB* b, size_t n) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto vb = mm512_uni_loadu_ps(b + i);
        mm512_uni_storeu_ps(a + i, vb);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto vb = mm256_uni_loadu_ps(b + i);
        mm256_uni_storeu_ps(a + i, vb);
    }
#endif
    for (; i < n; i++) {
        a[i] = b[i];
    }
}

template <typename T, typename T2>
void attn_memcpy_kernel(const ov::intel_cpu::PlainTensor& k_input,
                        const ov::intel_cpu::PlainTensor& v_input,
                        const ov::intel_cpu::PlainTensor& past_k_output,
                        const ov::intel_cpu::PlainTensor& past_v_output) {
    // For compatibility, all input_kvs are permuted to BHLS
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3], SV = v_input.m_dims[3];
    parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
        attn_copy(past_k_output.ptr<T2>(b, h, m, 0),
                  k_input.ptr<T>(b, h, m, 0),
                  S);
        attn_copy(past_v_output.ptr<T2>(b, h, m, 0),
                  v_input.ptr<T>(b, h, m, 0),
                  SV);
    });
}

static void attn_memcpy_kernel(const ov::intel_cpu::PlainTensor& k_input,
                               const ov::intel_cpu::PlainTensor& v_input,
                               const ov::intel_cpu::PlainTensor& past_k_output,
                               const ov::intel_cpu::PlainTensor& past_v_output) {
    // For compatibility, all input_kvs are permuted to BHLS
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3], SV = v_input.m_dims[3];
    parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
        std::memcpy(past_k_output.ptr_v(b, h, m, 0),
                    k_input.ptr_v(b, h, m, 0),
                    S * k_input.m_element_size);
        std::memcpy(past_v_output.ptr_v(b, h, m, 0),
                    v_input.ptr_v(b, h, m, 0),
                    SV * v_input.m_element_size);
    });
}

template <typename T, typename T2>
static void paged_attn_memcpy_kernel(const ov::intel_cpu::PlainTensor& k_input,
                                     const ov::intel_cpu::PlainTensor& v_input,
                                     const ov::intel_cpu::PlainTensor& past_k_output,
                                     const ov::intel_cpu::PlainTensor& past_v_output,
                                     const ov::intel_cpu::PlainTensor& slot_mapping) {
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3], SV = v_input.m_dims[3];
    size_t block_size = past_k_output.m_dims[2];
    parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
        auto slot = slot_mapping.ptr<int32_t>(b)[m];
        if (slot < 0) return;
        auto block_number = slot / block_size;
        auto block_offset = slot % block_size;
        attn_copy(past_k_output.ptr<T2>(block_number, h, block_offset, 0),
                  k_input.ptr<T>(b, h, m, 0),
                  S);
        attn_copy(past_v_output.ptr<T2>(block_number, h, block_offset, 0),
                  v_input.ptr<T>(b, h, m, 0),
                  SV);
    });
}

static void paged_attn_memcpy_kernel(const ov::intel_cpu::PlainTensor& k_input,
                                     const ov::intel_cpu::PlainTensor& v_input,
                                     const ov::intel_cpu::PlainTensor& past_k_output,
                                     const ov::intel_cpu::PlainTensor& past_v_output,
                                     const ov::intel_cpu::PlainTensor& slot_mapping) {
    size_t B = k_input.m_dims[0], H = k_input.m_dims[1], L1 = k_input.m_dims[2], S = k_input.m_dims[3], SV = v_input.m_dims[3];
    size_t block_size = past_k_output.m_dims[2];
    parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
        auto slot = slot_mapping.ptr<int32_t>(b)[m];
        if (slot < 0) return;
        auto block_number = slot / block_size;
        auto block_offset = slot % block_size;
        std::memcpy(past_k_output.ptr_v(block_number, h, block_offset, 0),
                    k_input.ptr_v(b, h, m, 0),
                    S * k_input.m_element_size);
        std::memcpy(past_v_output.ptr_v(block_number, h, block_offset, 0),
                    v_input.ptr_v(b, h, m, 0),
                    SV * v_input.m_element_size);
    });
}

void attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                 const ov::intel_cpu::PlainTensor& v_input,
                 const ov::intel_cpu::PlainTensor& past_k_output,
                 const ov::intel_cpu::PlainTensor& past_v_output) {
    if (past_k_output.get_precision() == k_input.get_precision()) {
        attn_memcpy_kernel(k_input, v_input, past_k_output, past_v_output);
    } else if (k_input.get_precision() == ov::element::f32 && past_k_output.get_precision() == ov::element::f16) {
        attn_memcpy_kernel<float, ov::float16>(k_input, v_input, past_k_output, past_v_output);
    } else if (k_input.get_precision() == ov::element::f32 && past_k_output.get_precision() == ov::element::bf16) {
        attn_memcpy_kernel<float, ov::bfloat16>(k_input, v_input, past_k_output, past_v_output);
    } else {
        OPENVINO_THROW("unsupport src type: ", k_input.get_precision(), ", dst type: ", past_k_output.get_precision(), " in attn_memcpy");
    }
}

void paged_attn_memcpy(const ov::intel_cpu::PlainTensor& k_input,
                       const ov::intel_cpu::PlainTensor& v_input,
                       const ov::intel_cpu::PlainTensor& past_k_output,
                       const ov::intel_cpu::PlainTensor& past_v_output,
                       const ov::intel_cpu::PlainTensor& slot_mapping) {
    if (past_k_output.get_precision() == k_input.get_precision()) {
        paged_attn_memcpy_kernel(k_input, v_input, past_k_output, past_v_output, slot_mapping);
    } else if (k_input.get_precision() == ov::element::f32 && past_k_output.get_precision() == ov::element::f16) {
        paged_attn_memcpy_kernel<float, ov::float16>(k_input, v_input, past_k_output, past_v_output, slot_mapping);
    } else if (k_input.get_precision() == ov::element::f32 && past_k_output.get_precision() == ov::element::bf16) {
        paged_attn_memcpy_kernel<float, ov::bfloat16>(k_input, v_input, past_k_output, past_v_output, slot_mapping);
    } else {
        OPENVINO_THROW("unsupport src type: ", k_input.get_precision(), ", dst type: ", past_k_output.get_precision(), " in paged_attn_memcpy");
    }
}

void attn_memcpy2d_kernel(void* src,
                          void* dst,
                          ov::element::Type src_type,
                          ov::element::Type dst_type,
                          size_t src_stride,
                          size_t dst_stride,
                          size_t width,
                          size_t height) {
    if (src_type == dst_type) {
        auto src_u8 = reinterpret_cast<uint8_t*>(src);
        auto dst_u8 = reinterpret_cast<uint8_t*>(dst);

        for (size_t j = 0; j < height; j++) {
            std::memcpy(dst_u8, src_u8, width * src_type.size());
            dst_u8 += dst_stride * src_type.size();
            src_u8 += src_stride * src_type.size();
        }
    } else if (src_type == ov::element::f32 && dst_type == ov::element::bf16) {
        auto src_f = reinterpret_cast<float*>(src);
        auto dst_f = reinterpret_cast<ov::bfloat16*>(dst);

        for (size_t j = 0; j < height; j++) {
            attn_copy<ov::bfloat16, float>(dst_f, src_f, width);
            dst_f += dst_stride;
            src_f += src_stride;
        }
    } else if (src_type == ov::element::f32 && dst_type == ov::element::f16) {
        auto src_f = reinterpret_cast<float*>(src);
        auto dst_f = reinterpret_cast<ov::float16*>(dst);

        for (size_t j = 0; j < height; j++) {
            attn_copy<ov::float16, float>(dst_f, src_f, width);
            dst_f += dst_stride;
            src_f += src_stride;
        }
    } else {
        OPENVINO_THROW("unsupport src type: ", src_type, ", dst type: ", dst_type, " in attn_memcpy2d_kernel");
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov