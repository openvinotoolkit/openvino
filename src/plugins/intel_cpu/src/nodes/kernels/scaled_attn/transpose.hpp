// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <type_traits>

#include "attn_quant_kernel.hpp"
#include "openvino/core/type/element_type.hpp"
#include "transpose_kernel.hpp"
#include "utils/general_utils.h"

namespace ov::Extensions::Cpu::XARCH {
using namespace ov::intel_cpu;

template <typename TDST, typename TSRC>
inline void transpose_tailx16_kernel(TDST* dst,
                                     TSRC* src,
                                     size_t n_cnt,
                                     size_t k_cnt,
                                     size_t dst_stride,
                                     size_t src_stride) {
    for (size_t i = 0; i < n_cnt; i++) {
        for (size_t j = 0; j < k_cnt; j++) {
            dst[j * dst_stride + i] = static_cast<TDST>(src[j + i * src_stride]);
        }
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<(none_of(SRC_PREC, ov::element::i8, ov::element::u8, ov::element::u4)), bool> = true>
void transpose_16NxK(TDST* dst,
                     void* src,
                     [[maybe_unused]] TDST* tmp,
                     const size_t N,
                     const size_t K,
                     const size_t block_size,
                     const size_t dst_stride,
                     const size_t src_stride,
                     [[maybe_unused]] const size_t group_size,
                     [[maybe_unused]] const bool quant_key_bychannel) {
    size_t k = 0;
    auto* src_ptr = reinterpret_cast<typename ov::element_type_traits<SRC_PREC>::value_type*>(src);
    for (size_t k = 0; k < K; k++) {
        memset(dst + k * dst_stride + N, 0, (block_size - N) * sizeof(TDST));
    }

    for (; k + 16 <= K; k += 16) {
        size_t n = 0;
        for (; n + 16 <= N; n += 16) {
            transpose_16x16_kernel(dst + n, src_ptr + n * src_stride, dst_stride, src_stride);
        }

        if (n < block_size) {
            transpose_tailx16_kernel(dst + n, src_ptr + n * src_stride, N - n, 16, dst_stride, src_stride);
        }

        dst += 16 * dst_stride;
        src_ptr += 16;
    }
    if (k < K) {
        size_t n = 0;
        for (; n + 16 <= N; n += 16) {
            transpose_16xK_kernel(dst + n, src_ptr + n * src_stride, K - k, dst_stride, src_stride);
        }

        if (n < block_size) {
            transpose_tailx16_kernel(dst + n, src_ptr + n * src_stride, N - n, K - k, dst_stride, src_stride);
        }
    }
}

#if defined(HAVE_AVX512F)
template <typename T,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<any_of(SRC_PREC, ov::element::bf16, ov::element::f16) &&
                                      (SRC_PREC == precision_of<T>::value),
                                  bool>::type = true>
void transpose_16NxK(T* dst,
                     T* src,
                     T* tmp,
                     const size_t N,
                     const size_t K,
                     const size_t block_size,
                     const size_t dst_stride,
                     const size_t src_stride,
                     const size_t group_size,
                     const bool quant_key_bychannel) {
    // will treat as uint32_t transpose
    auto s = reinterpret_cast<uint32_t*>(src);
    auto d = reinterpret_cast<uint32_t*>(dst);
    transpose_16NxK<uint32_t, ov::element::u32>(d,
                                                s,
                                                nullptr,
                                                N,
                                                K >> 1,
                                                block_size,
                                                dst_stride,
                                                src_stride >> 1,
                                                group_size,
                                                false);
}
#endif

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<any_of(SRC_PREC, ov::element::i8, ov::element::u8, ov::element::u4), bool> = true>
void transpose_16NxK(TDST* dst,
                     void* src,
                     TDST* tmp,
                     const size_t N,
                     const size_t K,
                     const size_t block_size,
                     const size_t dst_stride,
                     const size_t src_stride,
                     const size_t group_size,
                     const bool quant_key_bychannel) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto* s = reinterpret_cast<uint8_t*>(src);
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(SRC_PREC);
    constexpr size_t param_count = SRC_PREC == ov::element::i8 ? 1 : 2;
    auto t = tmp;
    // if group_size not set, the whole row is used as a group
    if (quant_key_bychannel) {
        if constexpr (any_of(SRC_PREC, ov::element::u8, ov::element::u4)) {
            auto* p_scales = reinterpret_cast<float*>(s);
            auto* p_zps = p_scales + K;
            s = s + sizeof(float) * 2 * K;
            attn_dequant_by_channel_kernel<TDST,
                                           SRC_PREC>(s, t, N, K, K / sub_byte_multiplier, src_stride, p_scales, p_zps);
        } else {
            OPENVINO_THROW("i8 doesn't support by-channel quantization");
        }
    } else {
        for (size_t n = 0; n < N; n++) {
            size_t src_offset = 0;
            size_t dst_offset = 0;
            while (dst_offset < K) {
                auto* params = reinterpret_cast<float*>(s + src_offset);
                attn_dequant_kernel<TDST, SRC_PREC>(s + src_offset + sizeof(float) * param_count,
                                                    t + dst_offset,
                                                    group_size,
                                                    params);
                src_offset += group_size / sub_byte_multiplier + sizeof(float) * param_count;
                dst_offset += group_size;
            }
            s += src_offset;
            t += src_stride;
        }
    }
    for (size_t n = N; n < block_size; n++) {
        memset(tmp + n * src_stride, 0, sizeof(TDST) * K);
    }
    transpose_16NxK<TDST, precision_of<TDST>::value>(dst,
                                                     tmp,
                                                     nullptr,
                                                     block_size,
                                                     K,
                                                     block_size,
                                                     dst_stride,
                                                     src_stride,
                                                     0,
                                                     false);
}

}  // namespace ov::Extensions::Cpu::XARCH