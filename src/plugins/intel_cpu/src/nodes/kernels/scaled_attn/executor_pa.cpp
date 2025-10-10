// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cpu/platform.hpp>
#include <cpu/x64/cpu_isa_traits.hpp>
#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#include "cpu_memory.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/element_type_traits.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "attn_memcpy.hpp"
#include "attn_quant.hpp"
#include "attn_quant_kernel.hpp"
#include "cache_rotation.hpp"
#include "executor_pa.hpp"
#include "executor_pa_common.hpp"
#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "paged_attn_kernel.hpp"
#include "sage_attn.hpp"
#include "softmax_kernel.hpp"
#include "transpose_kernel.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"
#if defined(OPENVINO_ARCH_X86_64)
#    include "nodes/kernels/x64/brgemm_kernel.hpp"
#elif defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE)
#    include "arm_sve.h"
#    include "nodes/kernels/aarch64/brgemm_kernel.hpp"
#    include "nodes/kernels/aarch64/sve_utils.hpp"
#    include "nodes/kernels/kai/kleidi_kernel.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

using namespace ov;
using namespace ov::intel_cpu;

// currently depends on brgemm which only support x64 or ARM SVE
#if defined(OPENVINO_ARCH_X86_64) || (defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE))

#    if defined(HAVE_AVX2) || defined(HAVE_AVX512F)

#        define prefetch_bytes(bytes, sel, advance, src) \
            {                                            \
                auto* p = reinterpret_cast<char*>(src);  \
                for (size_t i = 0; i < bytes; i += 64)   \
                    _mm_prefetch(p + i + advance, sel);  \
            }

#    else

#        define prefetch_bytes(bytes, sel, advance, src)

#    endif

// N must be multiple of 16
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
    // zero padding unsued blocks before transpose
    for (size_t n = N; n < block_size; n++) {
        memset(src_ptr + n * src_stride, 0, K * sizeof(typename ov::element_type_traits<SRC_PREC>::value_type));
    }
    for (; k + 16 <= K; k += 16) {
        for (size_t n = 0; n < block_size; n += 16) {
            transpose_16x16_kernel(dst + n, src_ptr + n * src_stride, dst_stride, src_stride);
        }

        dst += 16 * dst_stride;
        src_ptr += 16;
    }
    if (k < K) {
        for (size_t n = 0; n < block_size; n += 16) {
            transpose_16xK_kernel(dst + n, src_ptr + n * src_stride, K - k, dst_stride, src_stride);
        }
    }
}
#    if defined(HAVE_AVX512F)
template <typename T,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<any_of(SRC_PREC, ov::element::bf16, ov::element::f16) &&
                                      (SRC_PREC == precision_of<T>::value),
                                  bool>::type = true>
static void transpose_16NxK(T* dst,
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
                                                reinterpret_cast<uint32_t*>(0),
                                                N,
                                                K >> 1,
                                                block_size,
                                                dst_stride,
                                                src_stride >> 1,
                                                group_size,
                                                false);
}
#    endif

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
            static_assert(SRC_PREC == ov::element::i8, "i8 doesn't support by-channel quantization");
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
                                                     reinterpret_cast<TDST*>(0),
                                                     block_size,
                                                     K,
                                                     block_size,
                                                     dst_stride,
                                                     src_stride,
                                                     0,
                                                     false);
}

// dequant f16/u8 to float
template <typename T,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<SRC_PREC != ov::element::u8 && precision_of<T>::value == SRC_PREC, bool> = true>
static inline void dequant([[maybe_unused]] T* dst,
                           [[maybe_unused]] void* src,
                           [[maybe_unused]] const size_t N,
                           [[maybe_unused]] const size_t K,
                           [[maybe_unused]] const size_t block_size,
                           [[maybe_unused]] const size_t group_size,
                           [[maybe_unused]] const bool quant_bychannel) {
    // never called
    OPENVINO_THROW("dequant: should not be called.");
}
template <typename T, ov::element::Type_t SRC_PREC, std::enable_if_t<SRC_PREC == ov::element::f16, bool> = true>
static inline void dequant(float* dst,
                           void* src,
                           const size_t N,
                           const size_t K,
                           [[maybe_unused]] const size_t block_size,
                           [[maybe_unused]] const size_t group_size,
                           [[maybe_unused]] const bool quant_bychannel) {
    cvt_copy(dst, reinterpret_cast<ov::float16*>(src), 1, K * N, 0, 0);
    for (size_t i = N; i < block_size; i++) {
        memset(dst + i * K, 0, sizeof(float) * K);
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<any_of(SRC_PREC, ov::element::u4, ov::element::u8), bool> = true>
void dequant(TDST* dst,
             void* src,
             const size_t N,
             const size_t K,
             const size_t block_size,
             const size_t group_size,
             const bool quant_bychannel) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto* s = reinterpret_cast<uint8_t*>(src);
    const size_t params_offset = sizeof(float) * 2;
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(SRC_PREC);
    if (quant_bychannel) {
        auto* p_scales = reinterpret_cast<float*>(s);
        auto* p_zps = p_scales + K;
        s = s + sizeof(float) * 2 * K;
        if constexpr (any_of(SRC_PREC, ov::element::u8, ov::element::u4)) {
            attn_dequant_by_channel_kernel<TDST, SRC_PREC>(s, dst, N, K, K / sub_byte_multiplier, K, p_scales, p_zps);
        }
    } else {
        for (size_t n = 0; n < N; n++) {
            size_t src_offset = 0;
            size_t dst_offset = 0;
            while (dst_offset < K) {
                auto* params = reinterpret_cast<float*>(s + src_offset);
                attn_dequant_kernel<TDST, SRC_PREC>(s + src_offset + params_offset,
                                                    dst + dst_offset,
                                                    group_size,
                                                    params);
                src_offset += group_size / sub_byte_multiplier + params_offset;
                dst_offset += group_size;
            }
            s += src_offset;
            dst += K;
        }
    }
    for (size_t n = N; n < block_size; n++) {
        memset(dst, 0, sizeof(TDST) * K);
        dst += K;
    }
}

#    if defined(HAVE_AVX512F)
template <
    typename T,
    typename = typename std::enable_if_t<(std::is_same_v<T, ov::bfloat16> || std::is_same_v<T, ov::float16>), bool>>
static void pack_32x32_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    for (size_t i = 0; i < 16; i++) {
        auto a = _mm512_loadu_si512(src);  // [a1  a2  a3 a4 | a5  a6  a7 a8]   total 512-bits in 8 64bits unit
        auto b = _mm512_loadu_si512(src + src_stride);  // [b1  b2  b3 b4 | b5  b6  b7 b8]   total 512-bits
        a = _mm512_permutexvar_epi64(midx, a);          // [a1 a5 | a2 a6 | a3 a7 | a4 a8]
        b = _mm512_permutexvar_epi64(midx, b);          // [b1 b5 | b2 b6 | b3 b7 | b4 b8]
        auto B0 = _mm512_unpacklo_epi16(
            a,
            b);  // [ a1&b1  a2&b2   a3&b3   a4&b4] for each 128-bits lane, interleave word in low 64 bits
        auto B1 = _mm512_unpackhi_epi16(
            a,
            b);  // [ a5&b5  a6&b6   a7&b7   a8&b8] for each 128-bits lane, interleave word in high 64 bits
        _mm512_storeu_si512(dst, B0);
        _mm512_storeu_si512(dst + 32, B1);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template <
    typename T,
    typename = typename std::enable_if_t<(std::is_same_v<T, ov::bfloat16> || std::is_same_v<T, ov::float16>), bool>>
static void pack_32x16_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    for (size_t i = 0; i < 16; i++) {
        auto x =
            _mm256_loadu_si256(reinterpret_cast<__m256i*>(src));  // [a1  a2  a3 a4]   total 256-bits in 4 64bits unit
        auto y = _mm256_loadu_si256(reinterpret_cast<__m256i*>(src + src_stride));  // [b1  b2  b3 b4]   total 256-bits
        auto a = _mm512_castsi256_si512(x);
        auto b = _mm512_castsi256_si512(y);
        a = _mm512_permutexvar_epi64(midx, a);  // [a1 x | a2 x | a3 x | a4 x]
        b = _mm512_permutexvar_epi64(midx, b);  // [b1 x | b2 x | b3 x | b4 x]
        auto B0 = _mm512_unpacklo_epi16(a, b);
        _mm512_storeu_si512(dst, B0);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template <
    typename T,
    typename = typename std::enable_if_t<(std::is_same_v<T, ov::bfloat16> || std::is_same_v<T, ov::float16>), bool>>
static void pack_32xK_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride, size_t K) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    __mmask16 mask = (1 << K) - 1;
    for (size_t i = 0; i < 16; i++) {
        auto x = _mm256_maskz_loadu_epi16(mask, src);               // [a1  a2  a3 a4]   total 256-bits in 4 64bits unit
        auto y = _mm256_maskz_loadu_epi16(mask, src + src_stride);  // [b1  b2  b3 b4]   total 256-bits
        auto a = _mm512_castsi256_si512(x);
        auto b = _mm512_castsi256_si512(y);
        a = _mm512_permutexvar_epi64(midx, a);  // [a1 x | a2 x | a3 x | a4 x]
        b = _mm512_permutexvar_epi64(midx, b);  // [b1 x | b2 x | b3 x | b4 x]
        auto B0 = _mm512_unpacklo_epi16(a, b);
        _mm512_mask_storeu_epi32(dst, mask, B0);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if_t<precision_of<TDST>::value != ov::element::f32 &&
                                        any_of(SRC_PREC, ov::element::bf16, ov::element::f16),
                                    bool> = true>
static void pack_32NxK(TDST* dst,
                       void* src,
                       TDST* tmp,
                       const size_t N,
                       const size_t K,
                       const size_t block_size,
                       const size_t dst_stride,
                       const size_t src_stride,
                       const size_t group_size,
                       const bool quant_bychannel) {
    auto src_ptr = reinterpret_cast<typename ov::element_type_traits<SRC_PREC>::value_type*>(src);
    // zero padding unsued blocks before packing
    for (size_t n = N; n < block_size; n++) {
        memset(src_ptr + n * src_stride, 0, sizeof(TDST) * (K));
    }
    for (size_t n = 0; n < N; n += 32) {
        size_t k = 0;
        for (; k + 32 <= K; k += 32) {
            pack_32x32_kernel(dst + k * 2, src_ptr + k, dst_stride, src_stride);
        }
        if (k + 16 <= K) {
            pack_32x16_kernel(dst + k * 2, src_ptr + k, dst_stride, src_stride);
            k += 16;
        }
        if (k < K) {
            pack_32xK_kernel(dst + k * 2, src_ptr + k, dst_stride, src_stride, K - k);
        }

        dst += 32 * dst_stride;
        src_ptr += 32 * src_stride;
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<precision_of<TDST>::value != ov::element::f32 &&
                                      any_of(SRC_PREC, ov::element::u4, ov::element::u8),
                                  bool>::type = true>
static void pack_32NxK(TDST* dst,
                       void* src,
                       TDST* tmp,
                       const size_t N,
                       const size_t K,
                       const size_t block_size,
                       const size_t dst_stride,
                       const size_t src_stride,
                       const size_t group_size,
                       bool quant_bychannel) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized
    // feature(u8,idx_S)| The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto s = reinterpret_cast<uint8_t*>(src);
    auto t = tmp;
    // if group_size not set, the whole row is used as a group
    constexpr size_t sub_byte_multiplier = get_sub_byte_multiplier(SRC_PREC);
    if (quant_bychannel) {
        auto p_scales = reinterpret_cast<float*>(s);
        auto p_zps = p_scales + K;
        s = s + sizeof(float) * 2 * K;
        if constexpr (any_of(SRC_PREC, ov::element::u8, ov::element::u4)) {
            attn_dequant_by_channel_kernel<TDST,
                                           SRC_PREC>(s, t, N, K, K / sub_byte_multiplier, src_stride, p_scales, p_zps);
        }
    } else {
        for (size_t n = 0; n < N; n++) {
            size_t src_offset = 0;
            size_t dst_offset = 0;
            while (dst_offset < K) {
                auto params = reinterpret_cast<float*>(s + src_offset);
                attn_dequant_kernel<TDST, SRC_PREC>(s + (src_offset + sizeof(float) * 2),
                                                    t + dst_offset,
                                                    group_size,
                                                    params);
                src_offset += group_size / sub_byte_multiplier + sizeof(float) * 2;
                dst_offset += group_size;
            }
            s += src_offset;
            t += src_stride;
        }
    }
    for (size_t n = N; n < block_size; n++) {
        memset(tmp + n * src_stride, 0, sizeof(TDST) * (K));
    }
    pack_32NxK<TDST, precision_of<TDST>::value>(dst,
                                                tmp,
                                                reinterpret_cast<TDST*>(0),
                                                block_size,
                                                K,
                                                block_size,
                                                dst_stride,
                                                src_stride,
                                                group_size,
                                                quant_bychannel);
}
#    endif

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          std::enable_if_t<precision_of<TDST>::value == ov::element::f32, bool> = true>
static void pack_32NxK([[maybe_unused]] TDST* dst,
                       [[maybe_unused]] void* src,
                       [[maybe_unused]] TDST* tmp,
                       [[maybe_unused]] const size_t N,
                       [[maybe_unused]] const size_t K,
                       [[maybe_unused]] const size_t block_size,
                       [[maybe_unused]] const size_t dst_stride,
                       [[maybe_unused]] const size_t src_stride,
                       [[maybe_unused]] const size_t group_size,
                       [[maybe_unused]] const bool quant_bychannel) {
    // never called
    OPENVINO_THROW("pack_32NxK: should not be called.");
}

template <class T>
void fill_rotation_coefficients_from_lut(T* rotation_coefficients_block_data,
                                         const int32_t* rotation_deltas_block_data,
                                         size_t rotation_deltas_token_stride,
                                         const T* rotation_trig_lut,
                                         size_t block_size,
                                         size_t embedding_size) {
    size_t dst_offset = 0;
    for (size_t tok_idx = 0; tok_idx < block_size; tok_idx++) {
        size_t gather_idx = *(rotation_deltas_block_data + rotation_deltas_token_stride * tok_idx);
        size_t src_offset = gather_idx * embedding_size;
        std::memcpy(rotation_coefficients_block_data + dst_offset,
                    rotation_trig_lut + src_offset,
                    embedding_size * sizeof(T));
        dst_offset += embedding_size;
    }
}

template <ov::element::Type_t KEY_PREC>
void rotate_kv_cache(PlainTensor& key_cache,
                     const PlainTensor& rotated_block_indices,
                     const PlainTensor& rotation_deltas,
                     const PlainTensor& rotation_trig_lut,
                     PlainTensor& rotation_coefficients_scratch) {
    size_t num_blocks_in_total = key_cache.size(0);
    size_t num_heads = key_cache.size(1);  // H;
    size_t block_size = key_cache.size(2);
    size_t embedding_size = key_cache.size(3);  // S;

    size_t num_rotated_blocks = rotated_block_indices.size(0);
    auto* rotated_block_indices_data = rotated_block_indices.ptr<int32_t>();
    auto* rotation_trig_lut_data = rotation_trig_lut.ptr<float>();

    size_t rotation_deltas_token_stride = 0;
    size_t rotation_deltas_block_stride = 1;

    bool is_per_token = (rotation_deltas.shape()[1] == block_size);
    if (is_per_token) {
        rotation_deltas_token_stride = 1;
        rotation_deltas_block_stride = block_size;
    }

    for (size_t i = 0; i < num_rotated_blocks; i++) {
        size_t rotated_block_index = *(rotated_block_indices_data + i);
        OPENVINO_ASSERT(rotated_block_index < num_blocks_in_total);

        int32_t* rotation_deltas_block_data = rotation_deltas.ptr<int32_t>() + i * rotation_deltas_block_stride;

        auto* rotation_coefficient_block_data = rotation_coefficients_scratch.ptr<float>();
        fill_rotation_coefficients_from_lut(rotation_coefficient_block_data,
                                            rotation_deltas_block_data,
                                            rotation_deltas_token_stride,
                                            rotation_trig_lut_data,
                                            block_size,
                                            embedding_size);
        if constexpr (any_of(KEY_PREC, ov::element::i8, ov::element::u8, ov::element::u4)) {
            auto* cache_block_ptr = key_cache.ptr<uint8_t>(rotated_block_index);

            rotate_kv_cache_block(cache_block_ptr,
                                  rotation_coefficient_block_data,
                                  num_heads,
                                  block_size,
                                  embedding_size);
        } else {
            auto* cache_block_ptr =
                key_cache.ptr<typename ov::element_type_traits<KEY_PREC>::value_type>(rotated_block_index);
            rotate_kv_cache_block(cache_block_ptr,
                                  rotation_coefficient_block_data,
                                  num_heads,
                                  block_size,
                                  embedding_size);
        }
    }
}

struct ScoreAggregationInfo {
    int32_t score_offsets_aligned;  // tmp buffer offset for current block in the whole buffer
    int32_t score_offsets;          // dst buffer offset for output
    int32_t score_buf_num;          // tmp buffer number for current head
    int32_t kv_len_aligned;         // tmp buffer length for current block
};

template <typename DATA_TYPE, ov::element::Type_t KEY_PREC, ov::element::Type_t VALUE_PREC>
struct MHAHelper {
    // initialize once
    size_t H = 0UL;
    size_t S = 0UL;
    size_t SV = 0UL;
    size_t Hk = 0UL;
    size_t _h_each_group_len = 0UL;
    size_t _block_size = 0UL;
    size_t _nthr = 0UL;
    size_t _sliding_window = 0UL;
    float _d_scale = 0.0F;
    ov::Extensions::Cpu::PagedAttnQuantParams _params;
    size_t _new_score_stride = 0;
    bool AarchF16 = false;

    PlainTensor _weight;        // [nthr, H, 32, rnd_up(kv_len, block_size)], shared by first and second loop along bh
    PlainTensor _output;        // [nthr, 32, H, S], shared by first and second loop along bh
    PlainTensor _quantized_q;   // [H, batch_in_tokens, S + 1] (S + scale[f32]) / per-token
    PlainTensor _qk_scratch_a;  // [nthr, scratch_a_size]
    PlainTensor _qk_scratch_b;  // [B, rnd_up(kv_len, block_size), Hk, scratch_b_size]
    PlainTensor _wv_scratch_a;
    PlainTensor _wv_scratch_b;
    PlainTensor _alibi_lookup;
    PlainTensor _score_output;
    std::vector<size_t> _wsp;
    size_t _wsp_size_per_thread = 0;

    std::vector<std::shared_ptr<BrgemmKernel>> _qk_gemm;
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm;
    // will accumulate C buffer
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm_acc;
// second token
#    if defined(OPENVINO_ARCH_X86_64)
    std::shared_ptr<JitMatMulVecAMX> _gemv;
#    endif
    ov::element::Type _fastpath_valid_prec = ov::element::dynamic;
    // second token for bhl loop
    PlainTensor _weight_bhl;
    PlainTensor _output_bhl;

    std::vector<ScoreAggregationInfo> _score_infos;

    PlainTensor _block_rotation_coefficient_scratch;

    MHAHelper() {
        _weight.resize<float>({size_t{1}, size_t{1}, size_t{1}, size_t{1}});
    }

    explicit MHAHelper(const ov::Extensions::Cpu::PagedAttnQuantParams& params) : _params(params) {
        _weight.resize<float>({size_t{1}, size_t{1}, size_t{1}, size_t{1}});
    }

    void resize_temporary_weight_buffer(const size_t& h) {
        // resize temporary buffers, weight.size(3) will be aligned to block_size
        _weight.resize<float>({_nthr, h, _block_size, _new_score_stride});
    }

    void init(size_t H,
              size_t S,
              size_t SV,
              size_t Hk,
              size_t h_each_group_len,
              size_t block_size,
              size_t sliding_window,
              float d_scale,
              size_t B_token,
              size_t kv_len,
              bool init_alibi_lookup,
              bool init_rotation_coefficient_scratch) {
        // query shape: [B, H, L, S]
        // present_key shape: [block, H, 32, S]
        // Q*K': [M1, S] * [M2, S]'
        //   kernel: Q:[1~block_size, S] * K':[block_size, S]'
        //   aka: M:1~block_size, N:block_size, K:S
        // (Q*K')*V: [M1, M2] * [M2, S]
        //   kernel: (Q*K'):[1~block_size, block_size] * V:[block_size, S]
        //   aka: M:1~block_size, N:S, K:block_size
        // Because K and V are from cache, can use M2'=rnd_up(M2, block_size) to simplify logic
        auto in_type = precision_of<DATA_TYPE>::value;
        this->H = H;
        this->S = S;
        this->SV = SV;
        this->Hk = Hk;
        _h_each_group_len = h_each_group_len;
        _block_size = block_size;
        _nthr = static_cast<size_t>(parallel_get_max_threads());
        _sliding_window = sliding_window;
        _d_scale = d_scale;

#    if defined(OPENVINO_ARCH_ARM64)
        AarchF16 = any_of(precision_of<DATA_TYPE>::value, ov::element::f16);
#    endif
        auto prev_score_stride = _new_score_stride;
        auto want_score_stride = rnd_up(kv_len, _block_size);
        _new_score_stride = std::max(prev_score_stride, want_score_stride);
        // std::max(S, SV) here is to ensure by_channel quantize has enough buffer to use
        constexpr bool q_is_xf16 = any_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        if (_params.quant_key_bychannel || _params.quant_value_bychannel) {
            _output.resize<float>({_nthr, _block_size, H, std::max(S, SV)});
        } else {
            _output.resize<float>({_nthr, _block_size, H, SV});
        }

        // to ensure qk s8s8 s32 has enough buffer
        if (_params.is_sage_attn && H * std::max(S, SV) < _block_size) {
            _output.resize<float>({_nthr, _block_size, H, div_up(_block_size, H)});
        }

        if (_params.is_sage_attn) {
            _quantized_q.resize<int8_t>({_nthr, _block_size, S + sizeof(float)});
        }

        // TODO: kernel supports stride
        if (!AarchF16 && (_qk_gemm.empty() || prev_score_stride < _new_score_stride)) {
            _qk_gemm.resize(_block_size);
            _wv_gemm.resize(_block_size);
            _wv_gemm_acc.resize(_block_size);
            size_t wv_stride = q_is_xf16 ? _output.stride(1) : H * SV;
            for (size_t i = 0; i < _block_size; i++) {
                if (_params.is_sage_attn) {
                    // [M, K] x [N, K]^T
                    // LDB consider the offset of scale(f32)
                    // oneDNN brgemm has limitation with a_scale, it doesn't support per-row scale
                    // only enable b_scale here
#    if defined(OPENVINO_ARCH_X86_64)
                    // H * (S + sizeof(float))
                    _qk_gemm[i] = std::make_shared<BrgemmKernelQuantized>(
                        i + 1,                                                // M
                        _block_size,                                          // N
                        S,                                                    // K
                        (S + sizeof(float)),                                  // LDA
                        S + sizeof(float),                                    // LDB
                        _block_size,                                          // LDC
                        _new_score_stride,                                    // LDD
                        true,                                                 // b_transpose
                        ov::element::i8,                                      // in_prec
                        ov::element::f32,                                     // out_prec
                        ov::intel_cpu::BrgemmKernel::ScaleType::PER_CHANNEL,  // b_scale type
                        false);                                               // accumulate
#    endif
                } else {
                    _qk_gemm[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                                 _block_size,
                                                                 S,
                                                                 H * S,
                                                                 _block_size,
                                                                 _new_score_stride,
                                                                 false,
                                                                 in_type);
                }
                _wv_gemm[i] =
                    std::make_shared<BrgemmKernel>(i + 1,
                                                   SV,
                                                   _block_size,
                                                   // if it's bf16, the stride needs double due to reuse float buffer
                                                   (in_type == ov::element::Type_t::f32 ? 1 : 2) * _new_score_stride,
                                                   SV,
                                                   wv_stride,
                                                   false,
                                                   in_type);
                _wv_gemm_acc[i] =
                    std::make_shared<BrgemmKernel>(i + 1,
                                                   SV,
                                                   _block_size,
                                                   // if it's bf16, the stride needs double due to reuse float buffer
                                                   (in_type == ov::element::Type_t::f32 ? 1 : 2) * _new_score_stride,
                                                   SV,
                                                   wv_stride,
                                                   false,
                                                   in_type,
                                                   true);
            }

            // wsp is used to compute beta when K is blocked
            _wsp_size_per_thread = _wv_gemm[0]->get_wsp_size();
            _wsp.resize(_nthr * _wsp_size_per_thread);

            // allocate scratch a/b, notice get_scratch_a_size/get_scratch_b_size returns in bytes
            _qk_scratch_a.resize<DATA_TYPE>(
                {_nthr, _qk_gemm[_block_size - 1]->get_scratch_a_size() / sizeof(DATA_TYPE)});
            _wv_scratch_a.resize<DATA_TYPE>(
                {_nthr, _wv_gemm[_block_size - 1]->get_scratch_a_size() / sizeof(DATA_TYPE)});

#    if defined(OPENVINO_ARCH_X86_64)
            if ((S % 32 == 0) && (block_size % 16 == 0) && (S <= 32 * 6)) {
                if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16) &&
                    precision_of<DATA_TYPE>::value == ov::element::bf16 && KEY_PREC == ov::element::bf16 &&
                    VALUE_PREC == ov::element::bf16) {
                    _fastpath_valid_prec = ov::element::bf16;
                } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_fp16) &&
                           precision_of<DATA_TYPE>::value == ov::element::f16 && KEY_PREC == ov::element::f16 &&
                           VALUE_PREC == ov::element::f16) {
                    _fastpath_valid_prec = ov::element::f16;
                }
            }
            if (any_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16) && !_gemv) {
                _gemv = std::make_shared<JitMatMulVecAMX>(static_cast<int>(S),
                                                          static_cast<int>(block_size),
                                                          _fastpath_valid_prec);
            }
#    endif
        }

        if (init_alibi_lookup && (!_alibi_lookup || _alibi_lookup.m_dims[0] < kv_len)) {
            _alibi_lookup.resize<float>({kv_len * 2});
            for (size_t i = 0; i < _alibi_lookup.m_dims[0]; i++) {
                _alibi_lookup.ptr<float>()[i] = -static_cast<int>((_alibi_lookup.m_dims[0] - 1 - i));
            }
        }

        if (init_rotation_coefficient_scratch) {
            _block_rotation_coefficient_scratch.resize<DATA_TYPE>({_block_size, S});
        }
    }

    void init_reorder_buffers(size_t batch, size_t kv_len_in_blocks) {
        // append the scale after the tokens
        if (_params.is_sage_attn) {
            size_t padded_S = std::max((S + sizeof(float) / sizeof(int8_t)), rnd_up(S, 64));
            _qk_scratch_b.resize<int8_t>({batch, kv_len_in_blocks, Hk, _block_size * padded_S});
        } else {
            _qk_scratch_b.resize<DATA_TYPE>({batch, kv_len_in_blocks, Hk, _block_size * S});
        }
        if (AarchF16) {
            // It is required to keep kv_cache continuous in mem, as kleidi do not support accumulation
            _wv_scratch_b.resize<DATA_TYPE>({batch, Hk, kv_len_in_blocks, _block_size * rnd_up(SV, _block_size)});
        } else {
            _wv_scratch_b.resize<DATA_TYPE>({batch, kv_len_in_blocks, Hk, _block_size * rnd_up(SV, _block_size)});
        }
    }

    void init_score_buffers(const PlainTensor& past_lens,
                            const PlainTensor& subsequence_begins,
                            const PlainTensor& score_aggregation_window) {
        static constexpr int cache_line_size = dnnl::impl::cpu::platform::get_cache_line_size();
        auto seq_cout = static_cast<int32_t>(past_lens.m_dims[0]);
        _score_infos.resize(past_lens.m_dims[0]);
        int32_t total_kv_len_aligned = 0;
        int32_t total_kv_len = 0;
        for (int32_t i = 0; i < seq_cout; i++) {
            auto score_win_len = score_aggregation_window ? score_aggregation_window.ptr<int32_t>()[i] : 1;
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];

            // The score_aggregation_window may span multiple blocks, so need to allocate tmp buf for each block.
            // This will only occur in prefill.
            auto q_start_idx_score = q_len >= score_win_len ? q_len - score_win_len : 0;
            auto q_start_idx_score_block = q_start_idx_score / _block_size;
            auto q_end_idx_score_block = div_up(q_len, _block_size);
            auto tmp_buf_num = score_win_len ? q_end_idx_score_block - q_start_idx_score_block : 0;
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            // aligned to cache line to avoid false sharing
            auto kv_len_aligned = rnd_up(kv_len, cache_line_size / sizeof(float));
            _score_infos[i].score_offsets_aligned = total_kv_len_aligned;
            _score_infos[i].score_offsets = total_kv_len;
            _score_infos[i].score_buf_num = tmp_buf_num;
            _score_infos[i].kv_len_aligned = kv_len_aligned;
            total_kv_len_aligned += kv_len_aligned * tmp_buf_num;
            total_kv_len += kv_len;
        }

        _score_output.resize<float>({total_kv_len_aligned * H});
        parallel_for(H, [&](size_t h) {
            std::memset(_score_output.ptr<float>(h * total_kv_len_aligned), 0, total_kv_len_aligned * sizeof(float));
        });
    }

    // compute one block(such as 32 tokens) of query in M dimension: softmax(q_block*k')*v
    // all tensors such as query... have no batch dimension because batch dimension is varying
    //  query: [H, L, S]
    //  present_value: [block_number, H, 32, S]
    //  output_emb: [L, H * S]
    //  qk_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    //  wv_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    void exec_kernel_multiple(const PlainTensor& query,
                              const PlainTensor& present_value,
                              const PlainTensor& output_emb,
                              const PlainTensor& qk_scratch_b,
                              const PlainTensor& wv_scratch_b,
                              const int32_t* block_table,
                              size_t ithr,
                              size_t q_blk,
                              size_t hq_beg,
                              size_t hq_end,
                              size_t hk,
                              size_t q_len,
                              size_t cur_kv_len,
                              const PlainTensor& alibi_slopes,
                              float* score_output,
                              size_t q_start_idx_score,
                              const ScoreAggregationInfo* score_info_ptr) {
        auto q_start = q_blk * _block_size;
        auto q_end = std::min(q_start + _block_size, q_len);
        auto q_cnt = q_end - q_start;
        constexpr bool q_is_xf16 = any_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        constexpr bool q_cache_is_same = precision_of<DATA_TYPE>::value == VALUE_PREC;
        auto cur_kv_len_blocks = div_up(cur_kv_len, _block_size);
        for (size_t h = hq_beg; h < hq_end; h++) {
            auto* q_ptr = query.ptr<DATA_TYPE>(h, q_start, 0);
            if (_params.is_sage_attn) {
                // local_q shape [block_size, S + sizeof(float)]
                auto local_q = _quantized_q.slice(0, ithr, ithr + 1).reshape({_block_size, S + sizeof(float)});
                sage_attn_quantize_q<DATA_TYPE, ov::element::i8>(query, local_q, q_start, q_cnt, h);
            }
            auto* c_ptr = _weight.ptr<float>(ithr, h - hq_beg, 0, 0);
            // for each query block, loop through all key block
            // for blocks:
            // 1 0 0 0 ...
            // 1 1 0 0 ...
            // 1 1 1 0 ...
            // just computing the positions of 1 should be enough
            for (size_t k_blk = 0; k_blk < cur_kv_len_blocks; k_blk++) {
                if (_params.is_sage_attn) {
#    if defined(OPENVINO_ARCH_X86_64)
                    auto* q_ptr = _quantized_q.ptr<int8_t>(ithr);
                    auto* k_ptr = qk_scratch_b.ptr<int8_t>(k_blk, hk);
                    auto* temp_C = reinterpret_cast<int32_t*>(_output.ptr<float>(ithr, 0, 0, 0));
                    auto* scale_b = reinterpret_cast<float*>(qk_scratch_b.ptr<int8_t>(k_blk, hk, _block_size * S));
                    float* dst_f32 = c_ptr + k_blk * _block_size;
                    _qk_gemm[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                     q_ptr + sizeof(float),
                                                     k_ptr,
                                                     temp_C,
                                                     dst_f32,
                                                     scale_b,
                                                     _wsp.data() + ithr * _wsp_size_per_thread,
                                                     _qk_scratch_a ? _qk_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
#    endif
                } else {
                    auto* k_ptr = qk_scratch_b.ptr<DATA_TYPE>(k_blk, hk);
                    _qk_gemm[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                     q_ptr,
                                                     k_ptr,
                                                     c_ptr + k_blk * _block_size,
                                                     nullptr,
                                                     nullptr,
                                                     _wsp.data() + ithr * _wsp_size_per_thread,
                                                     _qk_scratch_a ? _qk_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
                }
            }

            for (size_t m = q_start; m < q_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = (cur_kv_len - q_cnt + (m - q_start) + 1);
                auto* score = _weight.ptr<float>(ithr, h - hq_beg, m - q_start);
                // dequantization of q matrix could be fused with _d_scale since softmax is done by row
                float revised_d_scale =
                    _params.is_sage_attn
                        ? _d_scale * reinterpret_cast<float*>(_quantized_q.ptr<int8_t>(ithr, m - q_start, 0))[0]
                        : _d_scale;
                if (_sliding_window) {
                    size_t start_idx = 0;
                    auto new_causal = ncausal;
                    float* alibi_lookup = nullptr;
                    if (ncausal > _sliding_window) {
                        start_idx = ncausal - _sliding_window;
                        new_causal = _sliding_window;
                    }
                    attn_softmax_kernel<float>(score + start_idx,
                                               reinterpret_cast<DATA_TYPE*>(score) + start_idx,
                                               revised_d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               new_causal,
                                               rnd_up(cur_kv_len, _block_size) - start_idx,
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value);

                    memset(score, 0, sizeof(DATA_TYPE) * start_idx);
                } else {
                    // alibi may available when _sliding_window is false
                    float* alibi_lookup = nullptr;
                    float alibi_slope = 0.F;
                    if (alibi_slopes) {
                        alibi_slope = alibi_slopes.ptr<float>()[h];
                        alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - ncausal;
                    }
                    attn_softmax_kernel<float>(score,
                                               reinterpret_cast<DATA_TYPE*>(score),
                                               revised_d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               ncausal,
                                               rnd_up(cur_kv_len, _block_size),
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value,
                                               alibi_slope);
                }
                if (score_output && m >= q_start_idx_score) {
                    auto* score_block_ptr =
                        score_output + h * score_info_ptr->kv_len_aligned * score_info_ptr->score_buf_num;
                    cvt_add(score_block_ptr,
                            score_block_ptr,
                            reinterpret_cast<DATA_TYPE*>(score),
                            1,
                            cur_kv_len,
                            0,
                            0,
                            0);
                }
            }

            // reuse float buffer, need to use float to compute offset
            auto* w_ptr = reinterpret_cast<DATA_TYPE*>(_weight.ptr<float>(ithr, h - hq_beg, 0, 0));
            float* fp32_out_ptr =
                q_is_xf16 ? _output.ptr<float>(ithr, 0, h, 0) : output_emb.ptr<float>(q_start, h * SV);

            // for each weight block, loop through all value block
            for (size_t v_blk = 0; v_blk < cur_kv_len_blocks; v_blk++) {
                DATA_TYPE* v_ptr = nullptr;
                if (q_is_xf16 || !q_cache_is_same) {
                    v_ptr = wv_scratch_b.ptr<DATA_TYPE>(v_blk, hk);
                } else {
                    v_ptr = present_value.ptr<DATA_TYPE>(block_table[v_blk], hk);
                }
                if (v_blk == 0) {
                    _wv_gemm[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                     w_ptr + v_blk * _block_size,
                                                     v_ptr,
                                                     fp32_out_ptr,
                                                     nullptr,
                                                     nullptr,
                                                     _wsp.data() + ithr * _wsp_size_per_thread,
                                                     _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
                } else {
                    _wv_gemm_acc[q_cnt - 1]->executeGemm(
                        q_cnt < _block_size,
                        w_ptr + v_blk * _block_size,
                        v_ptr,
                        fp32_out_ptr,
                        nullptr,
                        nullptr,
                        _wsp.data() + ithr * _wsp_size_per_thread,
                        _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
                }
            }
            if (q_is_xf16) {
                attn_memcpy2d_kernel(_output.ptr<float>(ithr, 0, h, 0),
                                     output_emb.ptr<DATA_TYPE>(q_start, h * SV),
                                     ov::element::f32,
                                     precision_of<DATA_TYPE>::value,
                                     _output.stride(1),
                                     output_emb.stride(0),
                                     SV,
                                     q_cnt);
            }
        }
    }
#    if defined(OPENVINO_ARCH_ARM64)
    // compute one block(such as 32 tokens) of query in M dimension: softmax(q_block*k')*v
    // all tensors such as query... have no batch dimension because batch dimension is varying
    //  query: [H, L, S]
    //  present_value: [block_number, H, 32, S]
    //  output_emb: [L, H * S]
    //  qk_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    //  wv_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    void exec_kernel_multiple_kai(const PlainTensor& query,
                                  const PlainTensor& present_value,
                                  const PlainTensor& output_emb,
                                  const PlainTensor& qk_scratch_b,
                                  const PlainTensor& wv_scratch_b,
                                  const int32_t* block_table,
                                  size_t ithr,
                                  size_t q_blk,
                                  size_t hq_beg,
                                  size_t hq_end,
                                  size_t hk,
                                  size_t q_len,
                                  size_t cur_kv_len,
                                  const PlainTensor& alibi_slopes,
                                  float* score_output,
                                  size_t q_start_idx_score,
                                  const ScoreAggregationInfo* score_info_ptr) {
        auto q_start = q_blk * _block_size;
        auto q_end = std::min(q_start + _block_size, q_len);
        auto q_cnt = q_end - q_start;
        constexpr bool q_is_xf16 = any_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        auto cur_kv_len_blocks = div_up(cur_kv_len, _block_size);
        auto _score_stride = _weight.stride_bytes(2) / 2;
        PlainTensor bias_wv, bias_qk;
        bias_wv.resize<float16_t>({SV});
        bias_qk.resize<float16_t>({_block_size});
        memset(bias_wv.ptr<float16_t>(0), 0, sizeof(DATA_TYPE) * SV);
        memset(bias_qk.ptr<float16_t>(0), 0, sizeof(DATA_TYPE) * _block_size);
        for (size_t h = hq_beg; h < hq_end; h++) {
            auto* q_ptr = query.ptr<DATA_TYPE>(h, q_start, 0);
            float* c_ptr = _weight.ptr<float>(ithr, h - hq_beg, 0, 0);
            // for each query block, loop through all key block
            // for blocks:
            // 1 0 0 0 ...
            // 1 1 0 0 ...
            // 1 1 1 0 ...
            // just computing the positions of 1 should be enough
            for (size_t k_blk = 0; k_blk < cur_kv_len_blocks; k_blk++) {
                auto* k_ptr = qk_scratch_b.ptr<DATA_TYPE>(k_blk, hk);
                auto* qk_out_ptr =
                    c_ptr +
                    (precision_of<DATA_TYPE>::value == ov::element::Type_t::f32 ? _block_size : _block_size / 2) *
                        k_blk;

                KleidiGemm qkKernel(q_cnt, _block_size, S, H * S, _block_size, _score_stride);
                PlainTensor packedB_k;
                packedB_k.resize<float16_t>({qkKernel.get_packed_rhs_size()});
                qkKernel.packB(reinterpret_cast<float16_t*>(k_ptr),
                               bias_qk.ptr<float16_t>(0),
                               packedB_k.ptr<float16_t>(0));
                qkKernel.executeGemm(q_ptr, packedB_k.ptr<float16_t>(0), qk_out_ptr);
            }

            for (size_t m = q_start; m < q_end; m++) {
                // apply softmax in f32 precision
                auto ncausal = (cur_kv_len - q_cnt + (m - q_start) + 1);
                auto soft_in = _weight.ptr<float>(ithr, h - hq_beg, m - q_start);
                auto score = _weight.ptr<float>(ithr, h - hq_beg, m - q_start);
                PlainTensor f32_cvt;
                if (q_is_xf16) {
                    f32_cvt.resize<float>({size_t{rnd_up(cur_kv_len, _block_size)}});
                    sve_utils::cvt_copy(f32_cvt.ptr<float>(0),
                                        reinterpret_cast<DATA_TYPE*>(score),
                                        rnd_up(cur_kv_len, _block_size));
                    soft_in = f32_cvt.ptr<float>(0);
                }
                if (_sliding_window) {
                    size_t start_idx = 0;
                    auto new_causal = ncausal;
                    float* alibi_lookup = nullptr;
                    if (ncausal > _sliding_window) {
                        start_idx = ncausal - static_cast<size_t>(_sliding_window);
                        new_causal = _sliding_window;
                    }
                    attn_softmax_kernel<float>(soft_in + start_idx,
                                               reinterpret_cast<DATA_TYPE*>(score) + start_idx,
                                               _d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               new_causal,
                                               rnd_up(cur_kv_len, _block_size) - start_idx,
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value);

                    memset(score, 0, sizeof(DATA_TYPE) * start_idx);
                } else {
                    // alibi may available when _sliding_window is false
                    float* alibi_lookup = nullptr;
                    float alibi_slope = 0.F;
                    if (alibi_slopes) {
                        alibi_slope = alibi_slopes.ptr<float>()[h];
                        alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - ncausal;
                    }
                    attn_softmax_kernel<float>(soft_in,
                                               reinterpret_cast<DATA_TYPE*>(score),
                                               _d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               ncausal,
                                               rnd_up(cur_kv_len, _block_size),
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value,
                                               alibi_slope);
                }
                if (score_output && m >= q_start_idx_score) {
                    // TODO: add sve opt code
                    auto score_block_ptr =
                        score_output + h * score_info_ptr->kv_len_aligned * score_info_ptr->score_buf_num;
                    cvt_add(score_block_ptr,
                            score_block_ptr,
                            reinterpret_cast<DATA_TYPE*>(score),
                            1,
                            cur_kv_len,
                            0,
                            0,
                            0);
                }
            }

            // reuse float buffer, need to use float to compute offset
            auto* w_ptr = reinterpret_cast<DATA_TYPE*>(_weight.ptr<float>(ithr, h - hq_beg, 0, 0));
            DATA_TYPE* out_ptr = output_emb.ptr<DATA_TYPE>(q_start, h * SV);
            DATA_TYPE* v_ptr;
            v_ptr = wv_scratch_b.ptr<DATA_TYPE>(hk, 0);
            PlainTensor packedB;
            KleidiGemm wvKernel(q_cnt, SV, _block_size * cur_kv_len_blocks, _score_stride, SV, H * SV);
            packedB.resize<float16_t>({wvKernel.get_packed_rhs_size()});
            wvKernel.packB(reinterpret_cast<float16_t*>(v_ptr), bias_wv.ptr<float16_t>(0), packedB.ptr<float16_t>(0));
            wvKernel.executeGemm(reinterpret_cast<float16_t*>(w_ptr), packedB.ptr<float16_t>(0), out_ptr);
        }
    }
#    endif

    // compute one token, loop along batch and head dimensions
    // all tensors such as query... have no batch dimension because batch dimension is varying
    //  query: [H, L, S]
    //  present_*: [block_number, H, 32, S]
    //  output_emb: [L, H * S]
    //  weight: [nthr, H, 32, rnd_up(kv_len, block_size)]
    //  output: [nthr, 32, H, S]
    void exec_kernel_one_bh(const PlainTensor& query,
                            const PlainTensor& present_key,
                            const PlainTensor& present_value,
                            const PlainTensor& output_emb,
                            const int32_t* block_table,
                            size_t ithr,
                            size_t hq_beg,
                            size_t hq_end,
                            size_t hk,
                            size_t q_len,
                            size_t cur_kv_len,
                            const PlainTensor& alibi_slopes,
                            float* score_output) {
#    if defined(OPENVINO_ARCH_X86_64)
        if (any_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16)) {
            _gemv->tile_config();
            for (size_t pk = 0, i = 0; pk < cur_kv_len; pk += _block_size, i++) {
                auto block_number = block_table[i];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hq_beg; h < hq_end; h++) {
                        (*_gemv)(
                            query.ptr<DATA_TYPE>(h, pq),
                            present_key.ptr<typename ov::element_type_traits<KEY_PREC>::value_type>(block_number, hk),
                            _weight.ptr<float>(ithr, h - hq_beg, pq) + pk);
                    }
                }
            }
            _gemv->tile_release();
        } else {
#    endif
            for (size_t pk = 0, i = 0; pk < cur_kv_len; pk += _block_size, i++) {
                auto block_number = block_table[i];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hq_beg; h < hq_end; h++) {
                        if constexpr (any_of(KEY_PREC, ov::element::i8, ov::element::u8, ov::element::u4)) {
                            dot_product_block_quantized<DATA_TYPE, KEY_PREC>(
                                query.ptr<DATA_TYPE>(h, pq),
                                present_key.ptr<uint8_t, KEY_PREC>(block_number, hk),
                                _weight.ptr<float>(ithr, h - hq_beg, pq) + pk,
                                S,
                                _params.quant_key_bychannel,
                                std::min(_block_size, cur_kv_len - pk),
                                _params.key_group_size);
                        } else {
                            dot_product_block<DATA_TYPE, KEY_PREC>(
                                query.ptr<DATA_TYPE>(h, pq),
                                present_key.ptr<typename ov::element_type_traits<KEY_PREC>::value_type>(block_number,
                                                                                                        hk),
                                _weight.ptr<float>(ithr, h - hq_beg, pq) + pk,
                                S,
                                std::min(_block_size, cur_kv_len - pk),
                                _params.key_group_size);
                        }
                    }
                }
            }
#    if defined(OPENVINO_ARCH_X86_64)
        }
#    endif

        for (size_t pq = 0; pq < q_len; pq++) {
            for (size_t h = hq_beg; h < hq_end; h++) {
                // apply attention mask & sofmax
                float* alibi_lookup = nullptr;
                float alibi_slope = 0.F;
                if (alibi_slopes) {
                    alibi_slope = alibi_slopes.ptr<float>()[h];
                    alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - cur_kv_len;
                }
                attn_softmax_kernel<float>(_weight.ptr<float>(ithr, h - hq_beg, pq),
                                           _weight.ptr<float>(ithr, h - hq_beg, pq),
                                           _d_scale,
                                           alibi_lookup,
                                           nullptr,
                                           nullptr,
                                           false,
                                           cur_kv_len,
                                           cur_kv_len,
                                           ov::element::f32,
                                           ov::element::f32,
                                           alibi_slope);
                if (score_output) {
                    // aligned to cache line to avoid false sharing
                    static constexpr int cache_line_size = dnnl::impl::cpu::platform::get_cache_line_size();
                    std::memcpy(score_output + h * rnd_up(cur_kv_len, cache_line_size / sizeof(float)),
                                _weight.ptr<float>(ithr, h - hq_beg, pq),
                                cur_kv_len * sizeof(float));
                }
            }
        }

        memset(_output.ptr<float>(ithr), 0, q_len * H * SV * sizeof(float));
        for (size_t pv = 0, i = 0; pv < cur_kv_len; pv += _block_size, i++) {
            auto block_number = block_table[i];
            for (size_t pq = 0; pq < q_len; pq++) {
                for (size_t h = hq_beg; h < hq_end; h++) {
                    if constexpr (any_of(VALUE_PREC, ov::element::u8, ov::element::u4)) {
                        attn_acc_value_block_quantized<uint8_t, VALUE_PREC>(
                            _output.ptr<float>(ithr, pq, h),
                            _weight.ptr<float>(ithr, h - hq_beg, pq) + pv,
                            present_value.ptr<uint8_t, VALUE_PREC>(block_number, hk),
                            SV,
                            _params.quant_value_bychannel,
                            std::min(_block_size, cur_kv_len - pv),
                            _params.value_group_size);
                    } else {
                        auto* v_ptr =
                            present_value.ptr<typename element_type_traits<VALUE_PREC>::value_type>(block_number, hk);
                        attn_acc_value_block<typename element_type_traits<VALUE_PREC>::value_type, VALUE_PREC>(
                            _output.ptr<float>(ithr, pq, h),
                            _weight.ptr<float>(ithr, h - hq_beg, pq) + pv,
                            v_ptr,
                            SV,
                            std::min(_block_size, cur_kv_len - pv),
                            _params.value_group_size);
                    }
                }
            }
        }
        // convert to dst
        for (size_t pq = 0; pq < q_len; pq++) {
            for (size_t h = hq_beg; h < hq_end; h++) {
                cvt_copy(output_emb.ptr<DATA_TYPE>(pq, h * SV), _output.ptr<float>(ithr, pq, h), 1, SV, 0, 0);
            }
        }
    }

    // compute one token, loop along batch, head dimensions and kv_len, it's special for very long kv_len with small
    // batch tokens. It will assume NO mixture execution of first and second token. all tensors such as query... have
    // batch dimension which is DIFFERENT from above
    //  query: [B, H, L, S]
    //  key_cache: [block_number, H, _block_size, S]
    //  value_cache: [block_number, H, _block_size, Sv]
    //  output_emb: [B, L, H * S]
    // 3 loops along batch, head, kv cache length dimensions
    void exec_loop_bhl(const PlainTensor& query,
                       PlainTensor& key_cache,
                       PlainTensor& value_cache,
                       const PlainTensor& output_emb,
                       const PlainTensor& output_score,
                       size_t max_context_len,
                       const PlainTensor& past_lens,
                       [[maybe_unused]] const PlainTensor& subsequence_begins,
                       const PlainTensor& block_indices,
                       const PlainTensor& block_indices_begins,
                       const PlainTensor& alibi_slopes,
                       const PlainTensor& score_aggregation_window) {
        auto B = past_lens.size(0);
        auto q_len = query.size(2);
        auto kv_len_in_blocks = div_up(max_context_len, _block_size);
        // aligned to cache line (64bytes=16*sizeof(float)) to avoid false sharing
        _weight_bhl.resize<float>({B, H, q_len, rnd_up(max_context_len, std::max(_block_size, size_t{16}))});

        // for small batches dynamic scheduler has notable overhead
        bool prefer_static_loop = false;
        // if less than 2 work items per thread, loop H
        bool loop_hk = B * kv_len_in_blocks * Hk > 2 * _nthr;
        if (B <= 32) {
            prefer_static_loop = true;
            // small batch and all batch size is same(like SDPA case)
            auto kv_len = past_lens.ptr<int32_t>()[0];
            for (size_t b = 1; b < B; b++) {
                if (past_lens.ptr<int32_t>()[b] != kv_len) {
                    prefer_static_loop = false;
                }
            }
        } else {
            // for bigger batch skip the test to save the cost
            prefer_static_loop = false;
        }
        auto get_h_params =
            [](bool loop_hk, size_t hx, size_t h_each_group_len, size_t& hq_beg, size_t& hq_end, size_t& hk) {
                if (loop_hk) {
                    hk = hx;
                    hq_beg = hk * h_each_group_len;
                    hq_end = (hk + 1) * h_each_group_len;
                } else {
                    hq_beg = hx;
                    hq_end = hx + 1;
                    hk = hx / h_each_group_len;
                }
            };
        auto loop_qk = [&](size_t b, size_t pk_in_blocks, size_t hx) {
            auto context_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            size_t hk = 0;
            size_t hq_beg = 0;
            size_t hq_end = 0;
            get_h_params(loop_hk, hx, _h_each_group_len, hq_beg, hq_end, hk);

            // kv_len must be valid
            auto pk = pk_in_blocks * _block_size;
            if (pk < context_len) {
                auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[b] + pk_in_blocks];
#    if defined(OPENVINO_ARCH_X86_64)
                if (any_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16)) {
                    _gemv->tile_config();
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = hq_beg; h < hq_end; h++) {
                            (*_gemv)(
                                query.ptr<DATA_TYPE>(b, h, pq),
                                key_cache.ptr<typename ov::element_type_traits<KEY_PREC>::value_type>(block_number, hk),
                                _weight_bhl.ptr<float>(b, h, pq) + pk);
                        }
                    }
                    _gemv->tile_release();
                } else {
#    endif
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = hq_beg; h < hq_end; h++) {
                            if constexpr (any_of(KEY_PREC, ov::element::i8, ov::element::u8, ov::element::u4)) {
                                dot_product_block_quantized<DATA_TYPE, KEY_PREC>(
                                    query.ptr<DATA_TYPE>(b, h, pq),
                                    key_cache.ptr<uint8_t, KEY_PREC>(block_number, hk),
                                    _weight_bhl.ptr<float>(b, h, pq) + pk,
                                    S,
                                    _params.quant_key_bychannel,
                                    std::min(_block_size, context_len - pk),
                                    _params.key_group_size);
                            } else {
                                dot_product_block<DATA_TYPE, KEY_PREC>(
                                    query.ptr<DATA_TYPE>(b, h, pq),
                                    key_cache.ptr<typename ov::element_type_traits<KEY_PREC>::value_type>(block_number,
                                                                                                          hk),
                                    _weight_bhl.ptr<float>(b, h, pq) + pk,
                                    S,
                                    std::min(_block_size, context_len - pk),
                                    _params.key_group_size);
                            }
                        }
                    }
#    if defined(OPENVINO_ARCH_X86_64)
                }
#    endif
            }
        };

        auto loop_softmax = [&](size_t b, size_t h, size_t pq) {
            auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            auto ncausal = cur_kv_len;
            // apply attention mask & sofmax
            float* alibi_lookup = nullptr;
            float alibi_slope = 0.F;
            if (alibi_slopes) {
                alibi_slope = alibi_slopes.ptr<float>()[h];
                alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - cur_kv_len;
            }
            attn_softmax_kernel<float>(_weight_bhl.ptr<float>(b, h, pq),
                                       _weight_bhl.ptr<float>(b, h, pq),
                                       _d_scale,
                                       alibi_lookup,
                                       nullptr,
                                       nullptr,
                                       false,
                                       ncausal,
                                       cur_kv_len,
                                       ov::element::f32,
                                       ov::element::f32,
                                       alibi_slope);
        };

        size_t h_dims = loop_hk ? Hk : H;
        if (prefer_static_loop) {
            parallel_for3d(B, kv_len_in_blocks, h_dims, loop_qk);
            parallel_for3d(B, H, q_len, loop_softmax);
        } else {
            parallel_for3d_dynamic(B, kv_len_in_blocks, h_dims, loop_qk);
            parallel_for3d_dynamic(B, H, q_len, loop_softmax);
        }

        if (output_score) {
            parallel_for2d_dynamic(B, q_len, [&](size_t b, size_t pq) {
                auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
                const auto score_win_len = score_aggregation_window ? score_aggregation_window.ptr<int32_t>()[b] : 1;
                auto* dst = output_score.ptr<float>() + _score_infos[b].score_offsets;
                if (score_win_len) {
                    auto* src = _weight_bhl.ptr<float>(b, 0, pq);
                    size_t src_stride = _weight_bhl.stride(2);
                    attn_reduce(dst, src, H, cur_kv_len, src_stride);
                } else {
                    std::memset(dst, 0, cur_kv_len * sizeof(float));
                }
            });
        }

        // attn_w * V
        _output_bhl.resize<float>({B, kv_len_in_blocks, H, q_len, SV});
        parallel_for3d(B, kv_len_in_blocks, H, [&](size_t b, size_t pv_in_blocks, size_t h) {
            memset(_output_bhl.ptr<float>(b, pv_in_blocks, h, 0, 0), 0, q_len * SV * sizeof(float));
        });

        auto loop_wk = [&](size_t b, size_t pv_in_blocks, size_t hx) {
            auto context_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            auto pv = pv_in_blocks * _block_size;
            size_t hk = 0;
            size_t hq_beg = 0;
            size_t hq_end = 0;
            get_h_params(loop_hk, hx, _h_each_group_len, hq_beg, hq_end, hk);

            // kv_len must be valid
            if (pv < context_len) {
                auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[b] + pv_in_blocks];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hq_beg; h < hq_end; h++) {
                        if constexpr (any_of(VALUE_PREC, ov::element::u8, ov::element::u4)) {
                            attn_acc_value_block_quantized<uint8_t, VALUE_PREC>(
                                _output_bhl.ptr<float>(b, pv_in_blocks, h, pq),
                                _weight_bhl.ptr<float>(b, h, pq) + pv,
                                value_cache.ptr<uint8_t, VALUE_PREC>(block_number, hk),
                                SV,
                                _params.quant_value_bychannel,
                                std::min(_block_size, context_len - pv),
                                _params.value_group_size);
                        } else {
                            auto* v_ptr =
                                value_cache.ptr<typename element_type_traits<VALUE_PREC>::value_type>(block_number, hk);
                            attn_acc_value_block<typename element_type_traits<VALUE_PREC>::value_type, VALUE_PREC>(
                                _output_bhl.ptr<float>(b, pv_in_blocks, h, pq),
                                _weight_bhl.ptr<float>(b, h, pq) + pv,
                                v_ptr,
                                SV,
                                std::min(_block_size, context_len - pv),
                                _params.value_group_size);
                        }
                    }
                }
            }
        };

        if (prefer_static_loop) {
            parallel_for3d(B, kv_len_in_blocks, loop_hk ? Hk : H, loop_wk);
        } else {
            parallel_for3d_dynamic(B, kv_len_in_blocks, loop_hk ? Hk : H, loop_wk);
        }

        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto* temp = _output_bhl.ptr<float>(b, 0, h, pq);
            size_t temp_stride = _output_bhl.stride(1);  // split with pv_in_blocks steps
            auto* dst = output_emb.ptr<DATA_TYPE>(b, pq, h * SV);
            attn_reduce(dst, temp, kv_len_in_blocks, SV, temp_stride);
        });
    }
};

template <typename DATA_TYPE, ov::element::Type_t KEY_PREC, ov::element::Type_t VALUE_PREC>
struct MHA {
    MHAHelper<DATA_TYPE, KEY_PREC, VALUE_PREC>& _helper;

    WorkItems _workitems;

    MHA(MHAHelper<DATA_TYPE, KEY_PREC, VALUE_PREC>& helper) : _helper(helper) {}

    // one loop to handle first and second tokens
    void exec_loop_mixed(const PlainTensor& q,
                         PlainTensor& k_cache,
                         const PlainTensor& v_cache,
                         const PlainTensor& output_emb,
                         const PlainTensor& output_score,
                         [[maybe_unused]] size_t max_context_len,
                         const PlainTensor& past_lens,
                         const PlainTensor& subsequence_begins,
                         const PlainTensor& block_indices,
                         const PlainTensor& block_indices_begins,
                         const PlainTensor& alibi_slopes,
                         const PlainTensor& score_aggregation_window) {
        auto Hk = v_cache.m_dims[1];

        constexpr bool q_is_xf16 = any_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        auto attn_work_count = _workitems.attn_work_size();
        auto reorder_work_count = _workitems.reorder_work_size();

        // buffer for transpose and repack
        _helper.init_reorder_buffers(_workitems.get_reorder_max_batch_size(),
                                     div_up(_workitems.get_reorder_max_kv_len(), _helper._block_size));

        // packed k, v
        parallel_for2d_dynamic(reorder_work_count, Hk, [&](size_t w, size_t hk) {
            constexpr bool q_cache_is_same = precision_of<DATA_TYPE>::value == VALUE_PREC;
            const auto& item = _workitems.get_reorder_work_item(w);
            const auto batch_in_seq = item.batch_in_seq;
            const auto batch_in_reorder = item.batch_in_reorder;
            const auto kv_block = item.kv_block_id;
            const auto block_number =
                block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[batch_in_seq] + kv_block];
            if (block_number < 0) {
                return;
            }

            const auto ithr = static_cast<size_t>(parallel_get_thread_num());
            const size_t valid_len = item.valid_block_len;
            if (_helper._params.is_sage_attn) {
#    if defined(OPENVINO_ARCH_X86_64)
                sage_attn_transpose_k(item,
                                      hk,
                                      _helper._block_size,
                                      _helper._qk_gemm[31],
                                      k_cache,
                                      _helper._qk_scratch_b);
#    endif
            } else {
                auto* k_ptr =
                    k_cache.ptr<typename ov::element_type_traits<KEY_PREC>::value_type, KEY_PREC>(block_number, hk);
                transpose_16NxK<DATA_TYPE, KEY_PREC>(
                    _helper._qk_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                    k_ptr,
                    _helper._output.template ptr<DATA_TYPE>(ithr),
                    valid_len,  // N
                    _helper.S,  // K
                    _helper._block_size,
                    _helper._block_size,                   // dst_stride
                    _helper.S,                             // src_stride
                    _helper._params.key_group_size,        // group_size
                    _helper._params.quant_key_bychannel);  // quant_by_channel
            }

            if (q_is_xf16) {
                auto* v_ptr =
                    v_cache.ptr<typename element_type_traits<VALUE_PREC>::value_type, VALUE_PREC>(block_number, hk);
#    if defined(OPENVINO_ARCH_ARM64)
                dequant<DATA_TYPE, VALUE_PREC>(
                    _helper._wv_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, hk, kv_block),
                    v_ptr,
                    valid_len,
                    _helper.SV,
                    _helper._block_size,
                    _helper._params.value_group_size,
                    _helper._params.quant_value_bychannel);
#    else
                    pack_32NxK<DATA_TYPE, VALUE_PREC>(
                        _helper._wv_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                        v_ptr,                                          // quantized data
                        _helper._output.template ptr<DATA_TYPE>(ithr),  // temp buffer hold dequantized data
                        valid_len,                                      // N may be smaller than block_size
                        _helper.SV,                                     // K
                        _helper._block_size,                            // block_size
                        rnd_up(_helper.SV, _helper._block_size),
                        _helper.SV,
                        _helper._params.value_group_size,
                        _helper._params.quant_value_bychannel);
#    endif
            } else {
                // need to decompress
                if constexpr (!q_cache_is_same) {
                    auto* v_ptr =
                        v_cache.ptr<typename ov::element_type_traits<VALUE_PREC>::value_type, VALUE_PREC>(block_number,
                                                                                                          hk);
                    dequant<DATA_TYPE, VALUE_PREC>(
                        _helper._wv_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                        v_ptr,
                        valid_len,
                        _helper.SV,
                        _helper._block_size,
                        _helper._params.value_group_size,
                        _helper._params.quant_value_bychannel);
                } else {
                    // zero padding unsued blocks
                    for (size_t n = valid_len; n < _helper._block_size; n++) {
                        auto* v_ptr = v_cache.ptr<typename ov::element_type_traits<VALUE_PREC>::value_type, VALUE_PREC>(
                            block_number,
                            hk,
                            n,
                            0);
                        memset(v_ptr,
                               0,
                               sizeof(typename ov::element_type_traits<VALUE_PREC>::value_type) * v_cache.m_dims[3]);
                    }
                }
            }
        });

        // loop along HK dimension: if mixed first/second token and elements count is enough, loop HK to reuse KV in the
        // CPU cache
        //    else if elements count is small, prefer to loop H to get more work to avoid thread imbalance
        bool loop_hk = _workitems.get_reorder_max_batch_size() == past_lens.m_dims[0] ||  // if only first token, loop H
                               attn_work_count * Hk <= 2 * _helper._nthr
                           ? false
                           : true;  // or less than 2 work items per thread, loop H
        auto weight_h = loop_hk ? _helper.H / Hk : 1;
        _helper.resize_temporary_weight_buffer(weight_h);
        // attn_work_count num_sub_seq
        parallel_for2d_dynamic(attn_work_count, loop_hk ? Hk : _helper.H, [&](size_t w, size_t hx) {
            size_t hk = 0;
            size_t hq_beg = 0;
            size_t hq_end = 0;
            if (loop_hk) {
                hk = hx;
                hq_beg = hk * _helper._h_each_group_len;
                hq_end = (hk + 1) * _helper._h_each_group_len;
            } else {
                hq_beg = hx;
                hq_end = hx + 1;
                hk = hx / _helper._h_each_group_len;
            }

            const auto& item = _workitems.get_attn_work_item(w);
            const auto batch_in_seq = item.batch_in_seq;
            const auto batch_in_token = subsequence_begins.ptr<int32_t>()[batch_in_seq];
            const auto q_len = static_cast<size_t>(item.q_len);
            const auto ithr = static_cast<size_t>(parallel_get_thread_num());

            if (q_len == 1) {
                const auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[batch_in_seq]) + 1;
                float* score_output = nullptr;
                if (output_score) {
                    const auto score_win_len =
                        score_aggregation_window ? score_aggregation_window.ptr<int32_t>()[batch_in_seq] : 1;
                    if (score_win_len) {
                        auto score_offset = _helper._score_infos[batch_in_seq].score_offsets_aligned;
                        score_output = _helper._score_output.template ptr<float>() + score_offset * _helper.H;
                    }
                }

                _helper.exec_kernel_one_bh(
                    q.slice(0, batch_in_token, batch_in_token),
                    k_cache,
                    v_cache,
                    output_emb.slice(0, batch_in_token, batch_in_token),
                    block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                    ithr,
                    hq_beg,
                    hq_end,
                    hk,
                    1UL,
                    cur_kv_len,
                    alibi_slopes,
                    score_output);
            } else {
                const auto batch_in_reorder = item.batch_in_reorder;
                const auto q_blk = item.q_block_id;
                const auto q_cnt = std::min(_helper._block_size, q_len - q_blk * _helper._block_size);
                const auto cur_kv_len =
                    static_cast<size_t>(past_lens.ptr<int32_t>()[batch_in_seq]) + q_blk * _helper._block_size + q_cnt;
                float* score_output = nullptr;
                size_t q_start_idx_score = 0;
                ScoreAggregationInfo* score_info_ptr = nullptr;
                if (output_score) {
                    const auto score_win_len =
                        score_aggregation_window
                            ? static_cast<size_t>(score_aggregation_window.ptr<int32_t>()[batch_in_seq])
                            : 1;
                    if (score_win_len) {
                        q_start_idx_score = q_len >= score_win_len ? q_len - score_win_len : 0;
                        if (q_cnt + q_blk * _helper._block_size > q_start_idx_score) {
                            score_info_ptr = &_helper._score_infos[batch_in_seq];
                            auto score_offset =
                                score_info_ptr->score_offsets_aligned * _helper.H +
                                (q_blk - q_start_idx_score / _helper._block_size) * score_info_ptr->kv_len_aligned;
                            score_output = _helper._score_output.template ptr<float>() + score_offset;
                        }
                    }
                }

                PlainTensor sub_query;

                sub_query.resize({q_len, _helper.H, _helper.S}, q.ptr<DATA_TYPE>(batch_in_token));
                // physical layout (B_in_tokens, H, S)
                sub_query = sub_query.permute({1, 0, 2});
#    if defined(OPENVINO_ARCH_ARM64)
                if constexpr (q_is_xf16) {
                    _helper.exec_kernel_multiple_kai(
                        sub_query,
                        v_cache,
                        output_emb.slice(0, batch_in_token, batch_in_token + q_len)
                            .reshape({q_len, _helper.H * _helper.SV}),
                        _helper._qk_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                        _helper._wv_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                        block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                        ithr,
                        q_blk,
                        hq_beg,
                        hq_end,
                        hk,
                        q_len,
                        cur_kv_len,
                        alibi_slopes,
                        score_output,
                        q_start_idx_score,
                        score_info_ptr);
                } else {
                    _helper.exec_kernel_multiple(
                        sub_query,
                        v_cache,
                        output_emb.slice(0, batch_in_token, batch_in_token + q_len)
                            .reshape({q_len, _helper.H * _helper.SV}),
                        _helper._qk_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                        _helper._wv_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                        block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                        ithr,
                        q_blk,
                        hq_beg,
                        hq_end,
                        hk,
                        q_len,
                        cur_kv_len,
                        alibi_slopes,
                        score_output,
                        q_start_idx_score,
                        score_info_ptr);
                }
#    else
                _helper.exec_kernel_multiple(
                    sub_query,
                    v_cache,
                    output_emb.slice(0, batch_in_token, batch_in_token + q_len)
                        .reshape({q_len, _helper.H * _helper.SV}),
                    _helper._qk_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                    _helper._wv_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                    block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                    ithr,
                    q_blk,
                    hq_beg,
                    hq_end,
                    hk,
                    q_len,
                    cur_kv_len,
                    alibi_slopes,
                    score_output,
                    q_start_idx_score,
                    score_info_ptr);
#    endif
            }
        });
        if (output_score) {
            parallel_for2d_dynamic(past_lens.m_dims[0], 1, [&](size_t b, [[maybe_unused]] size_t pq) {
                auto seq_len = static_cast<size_t>(subsequence_begins.ptr<int32_t>()[b + 1] -
                                                   subsequence_begins.ptr<int32_t>()[b]);
                auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + seq_len;
                const auto score_win_len = score_aggregation_window ? score_aggregation_window.ptr<int32_t>()[b] : 1;
                const auto& score_info = _helper._score_infos[b];
                auto dst_offset = score_info.score_offsets;
                auto* dst = output_score.ptr<float>() + dst_offset;
                if (score_win_len) {
                    auto* src =
                        _helper._score_output.template ptr<float>() + score_info.score_offsets_aligned * _helper.H;
                    size_t src_stride = score_info.kv_len_aligned;
                    attn_reduce(dst, src, _helper.H * score_info.score_buf_num, cur_kv_len, src_stride);
                } else {
                    std::memset(dst, 0, cur_kv_len * sizeof(float));
                }
            });
        }
    }

    // Q, K, V is ready, do attention
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    PlainTensor& output_score,
                    size_t max_context_len,
                    const PlainTensor& past_lens,
                    const PlainTensor& subsequence_begins,
                    const PlainTensor& block_indices,
                    const PlainTensor& block_indices_begins,
                    const PlainTensor& alibi_slopes,
                    const PlainTensor& score_aggregation_window) {
        _workitems
            .reset(query, past_lens, subsequence_begins, block_indices, block_indices_begins, _helper._block_size);
        if (output_score) {
            _helper.init_score_buffers(past_lens, subsequence_begins, score_aggregation_window);
        }

        auto nthr = static_cast<size_t>(parallel_get_max_threads());

        if (past_lens.m_dims[0] >= nthr || _workitems.get_reorder_max_batch_size() > 0) {
            exec_loop_mixed(query,
                            present_key,
                            present_value,
                            output_emb,
                            output_score,
                            max_context_len,
                            past_lens,
                            subsequence_begins,
                            block_indices,
                            block_indices_begins,
                            alibi_slopes,
                            score_aggregation_window);
        } else {
            _helper.exec_loop_bhl(query,
                                  present_key,
                                  present_value,
                                  output_emb,
                                  output_score,
                                  max_context_len,
                                  past_lens,
                                  subsequence_begins,
                                  block_indices,
                                  block_indices_begins,
                                  alibi_slopes,
                                  score_aggregation_window);
        }
    }
};

template <typename DATA_TYPE, ov::element::Type_t KEY_PREC, ov::element::Type_t VALUE_PREC>
struct AttentionExecutor : public PagedAttentionExecutor {
    MHAHelper<DATA_TYPE, KEY_PREC, VALUE_PREC> _helper;
    MHA<DATA_TYPE, KEY_PREC, VALUE_PREC> _kernel;
    PlainTensor _slot_mapping;

    AttentionExecutor() : _kernel(_helper) {}

    explicit AttentionExecutor(const ov::Extensions::Cpu::PagedAttnQuantParams& params)
        : _helper(MHAHelper<DATA_TYPE, KEY_PREC, VALUE_PREC>(params)),
          _kernel(_helper) {}

    void init(const std::vector<MemoryPtr>& inputs,
              const std::vector<MemoryPtr>& outputs,
              PlainTensor& q,
              PlainTensor& k,
              PlainTensor& v,
              PlainTensor& k_cache,
              PlainTensor& v_cache,
              PlainTensor& past_lens,
              PlainTensor& subsequence_begins,
              PlainTensor& block_indices,
              PlainTensor& block_indices_begins,
              float& scale,
              size_t& sliding_window,
              PlainTensor& alibi_slopes,
              size_t& max_context_len,
              PlainTensor& score_aggregation_window,
              PlainTensor& rotated_block_indices,
              PlainTensor& rotation_deltas,
              PlainTensor& rotation_trig_lut,
              PlainTensor& xattention_threshold,
              int32_t& xattention_block_size,
              int32_t& xattention_stride,
              PlainTensor& sinks,
              PlainTensor& output_emb,
              PlainTensor& output_score) {
        q.reset(inputs[ID_Q]);  // [B_token, H * S]
        k.reset(inputs[ID_K]);
        v.reset(inputs[ID_V]);
        k_cache.reset(inputs[ID_KCACHE]);                             // [NUM_BLOCKS, H, 32, S]
        v_cache.reset(inputs[ID_VCACHE]);                             // [NUM_BLOCKS, H, 32, S]
        past_lens.reset(inputs[ID_PAST_LENS]);                        // [B_seq]
        subsequence_begins.reset(inputs[ID_SUBSEQUENCE_BEGINS]);      // [B_seq+1]
        block_indices.reset(inputs[ID_BLOCK_INDICES]);                // [num_blocks]
        block_indices_begins.reset(inputs[ID_BLOCK_INDICES_BEGINS]);  // [B_seq+1]
        scale = *inputs[ID_SCALE]->getDataAs<float>();
        sliding_window = static_cast<size_t>(*inputs[ID_SLIDING_WINDOW]->getDataAs<int32_t>());
        if (!inputs[ID_ALIBI_SLOPES]->getShape().hasZeroDims()) {
            alibi_slopes.reset(inputs[ID_ALIBI_SLOPES]);
        }
        max_context_len = static_cast<size_t>(*inputs[ID_MAX_CONTEXT_LEN]->getDataAs<int32_t>());

        if (!inputs[ID_SCORE_AGGREGATION_WINDOW]->getShape().hasZeroDims()) {
            score_aggregation_window.reset(inputs[ID_SCORE_AGGREGATION_WINDOW]);  // [B_seq]
        }

        size_t inputs_size = inputs.size();
        OPENVINO_ASSERT(inputs_size == 21);
        if (!inputs[ID_ROTATED_BLOCK_INDICES]->getShape().hasZeroDims()) {
            rotated_block_indices.reset(inputs[ID_ROTATED_BLOCK_INDICES]);  // [num_blocks]
        }
        if (!inputs[ID_ROTATION_DELTAS]->getShape().hasZeroDims()) {
            rotation_deltas.reset(inputs[ID_ROTATION_DELTAS]);  // [num_blocks,  block_size (32) || 1]
        }
        if (!inputs[ID_ROTATION_TRIG_LUT]->getShape().hasZeroDims()) {
            rotation_trig_lut.reset(
                inputs[ID_ROTATION_TRIG_LUT]);  // [max_context_len * embedding_size], row-major layout
        }

        if (!inputs[ID_XATTENTION_THRESHOLD]->getShape().hasZeroDims()) {
            xattention_threshold.reset(inputs[ID_XATTENTION_THRESHOLD]);  // [B_seq]
        }
        xattention_stride = *inputs[ID_XATTENTION_STRIDE]->getDataAs<int32_t>();
        xattention_block_size = *inputs[ID_XATTENTION_BLOCK_SIZE]->getDataAs<int32_t>();

        if (!inputs[ID_SINKS]->getShape().hasZeroDims()) {
            sinks.reset(inputs[ID_SINKS]);  // [1, 64, 1, 1]
        }

        output_emb.reset(outputs[0]);
        if (outputs.size() == 2) {
            output_score.reset(outputs[1]);
        }

        auto B_token = q.size(0);
        auto Hk = k_cache.size(1);
        /* The layout for kv cache:

           by-token, quantized by S(token dims) group_num = N
           N * f32(scale + zp)|group_0|group_1|...|group_N
           adjusted_S = S + N * f32(scale + zp) * sub_byte_multiplier

           by channel, quantized by channel [block_size, S], group_num = block_size
           adjusted block consists 3 parts
           f32[scale, S]
           f32[zp, S]
           u8[block_size, S]
           adjusted key cache block u8[block_size + 2 * sizeof(float) * sub_byte_multiplier, S]
        */

        const size_t key_sub_byte_multiplier = get_sub_byte_multiplier(k_cache.get_precision());
        const size_t value_sub_byte_multiplier = get_sub_byte_multiplier(v_cache.get_precision());
        const size_t params_count = k_cache.get_precision() == ov::element::i8 ? 1 : 2;
        const size_t key_params_size = sizeof(float) * params_count * key_sub_byte_multiplier;
        // u4 needs scale + zp. s4 needs scale.
        const size_t param_size =
            any_of(v_cache.get_precision(), ov::element::u4, ov::element::u8) ? sizeof(float) * 2 : sizeof(float);
        const size_t value_params_size = param_size * value_sub_byte_multiplier;
        size_t key_group_num =
            _helper._params.key_group_size ? k_cache.size(3) / (_helper._params.key_group_size + key_params_size) : 1;
        size_t value_group_num = _helper._params.value_group_size
                                     ? v_cache.size(3) / (_helper._params.value_group_size + value_params_size)
                                     : 1;

        // check by_token_dims parameter of value cache
        OPENVINO_ASSERT(value_group_num != 0U || !v_cache.get_precision().is_integral(),
                        "PagedAttn value cache gets wrong group_size, ",
                        _helper._params.value_group_size,
                        " should be smaller than token_dims");
        size_t S = 0;
        // check parameter of quantized key cache
        if (k_cache.get_precision().is_integral()) {
            if (_helper._params.quant_key_bychannel) {
                S = k_cache.size(3);
            } else {
                OPENVINO_ASSERT(key_group_num,
                                "PagedAttn key cache gets wrong group_size, ",
                                _helper._params.key_group_size,
                                " should be smaller than token_dims");
                S = k_cache.size(3) - key_params_size * key_group_num;
                _helper._params.key_group_size = _helper._params.key_group_size ? _helper._params.key_group_size : S;
            }
        } else {
            S = k_cache.size(3);
        }

        size_t SV = 0;
        // check parameter of quantized value cache
        if (v_cache.get_precision().is_integral()) {
            if (_helper._params.quant_value_bychannel) {
                SV = v_cache.size(3);
            } else {
                OPENVINO_ASSERT(value_group_num,
                                "PagedAttn value cache gets wrong group_size, ",
                                _helper._params.value_group_size,
                                " should be smaller than token_dims");
                SV = v_cache.size(3) - value_params_size * value_group_num;
                _helper._params.value_group_size =
                    _helper._params.value_group_size ? _helper._params.value_group_size : SV;
            }
        } else {
            SV = v_cache.size(3);
        }
        auto block_size = (_helper._params.quant_key_bychannel && k_cache.get_precision().is_integral())
                              ? (k_cache.size(2) - key_params_size)
                              : k_cache.size(2);
        auto H = q.size(1) / S;
        auto h_each_group_len = 1;
        if (Hk != H) {
            h_each_group_len = H / Hk;
        }
        auto B_seq = past_lens.size(0);
        q.assert_dims({B_token, H * S});
        k.assert_dims({B_token, Hk * S});
        v.assert_dims({B_token, Hk * SV});
        q = q.reshape({B_token, H, 1, S});
        k = k.reshape({B_token, Hk, 1, S});
        v = v.reshape({B_token, Hk, 1, SV});
        if (k_cache.get_precision().is_integral()) {
            if (_helper._params.quant_key_bychannel) {
                k_cache.assert_dims({0, Hk, block_size + key_params_size, S}, true);
            } else {
                k_cache.assert_dims({0, Hk, block_size, S + key_params_size * key_group_num}, true);
            }

        } else {
            k_cache.assert_dims({0, Hk, block_size, S}, true);
        }
        if (v_cache.get_precision().is_integral()) {
            if (_helper._params.quant_value_bychannel) {
                v_cache.assert_dims({0, Hk, block_size + value_params_size, SV}, true);
            } else {
                v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV + value_params_size * value_group_num},
                                    true);
            }
        } else {
            v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV});
        }
        past_lens.assert_dims({B_seq});
        subsequence_begins.assert_dims({B_seq + 1});
        block_indices.assert_dims({0}, true);
        block_indices_begins.assert_dims({B_seq + 1});
        if (scale == 0.0F) {
            scale = 1.0F / sqrt(S);
        }
        if (alibi_slopes) {
            alibi_slopes.assert_dims({H});
        }

        if (score_aggregation_window) {
            score_aggregation_window.assert_dims({B_seq});
        }

        bool init_rotation_coefficient_scratch = false;
        if (rotated_block_indices) {
            // Only K entries are needed to be rotated, since position is encoded at the Q^T @ (effective_RoPE_matrix) @
            // K matrix multiplication
            rotation_deltas.assert_dims({rotated_block_indices.size(0), 0}, /* special_zero = */ true);
            OPENVINO_ASSERT(any_of(rotation_deltas.shape()[1], 1U, block_size));  // per-block or per-token granularity
            rotation_trig_lut.assert_dims({0, S}, /* special_zero = */ true);
            init_rotation_coefficient_scratch = true;
        }
        if (xattention_threshold) {
            xattention_threshold.assert_dims({B_seq});
            OPENVINO_ASSERT(xattention_block_size > 0);
            OPENVINO_ASSERT(xattention_stride > 0);
            // TODO: add assertions on the block size and stride limitations as defined by the
            // block sparse operation and importance score computation impls
        }

        if (sinks) {
            sinks.assert_dims({1, H, 1, 1});
        }

        output_emb.assert_dims({B_token, H * SV});
        output_emb = output_emb.reshape({B_token, 1, H * SV});

        // TODO: enable block_size to be multiple of 32
        OPENVINO_ASSERT(block_size == 32, "CPU: block size must be 32, current: ", block_size);

        _helper.init(H,
                     S,
                     SV,
                     Hk,
                     h_each_group_len,
                     block_size,
                     sliding_window,
                     scale,
                     B_token,
                     max_context_len,
                     static_cast<bool>(alibi_slopes),
                     init_rotation_coefficient_scratch);
    }

    void concat_pastkv(const PlainTensor& k,
                       const PlainTensor& v,
                       PlainTensor& k_cache,
                       PlainTensor& v_cache,
                       const PlainTensor& past_lens,
                       const PlainTensor& subsequence_begins,
                       const PlainTensor& block_indices,
                       const PlainTensor& block_indices_begins) {
        auto B_token = k.size(0);
        _slot_mapping.resize<int32_t>({B_token});
        size_t idx = 0;
        for (size_t i = 0; i < past_lens.size(0); i++) {
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            auto block_number_start = block_indices_begins.ptr<int32_t>()[i];
            auto block_offset_start = kv_len - q_len;
            for (int32_t j = 0; j < q_len; j++) {
                auto block_offset = block_offset_start + j;
                auto block_number =
                    block_indices.ptr<int32_t>()[block_number_start + block_offset / _helper._block_size];
                _slot_mapping.ptr<int32_t>()[idx++] =
                    block_number * _helper._block_size + block_offset % _helper._block_size;
            }
        }

        if constexpr (any_of(KEY_PREC, ov::element::i8, ov::element::u8, ov::element::u4)) {
            // slot_mapping could only be used for per token quantization
            // by_channel needs all data to calculation block info.
            QuantizeParam quant_params;
            quant_params.quant_key_by_channel = _helper._params.quant_key_bychannel;
            quant_params.quant_value_by_channel = _helper._params.quant_value_bychannel;
            quant_params.key_group_size = _helper._params.key_group_size;
            quant_params.value_group_size = _helper._params.value_group_size;
            quant_params.is_sage_attn = _helper._params.is_sage_attn;

            paged_attn_quantkv(k,
                               v,
                               k_cache,
                               v_cache,
                               past_lens,
                               subsequence_begins,
                               block_indices,
                               block_indices_begins,
                               _slot_mapping,
                               _helper._output,
                               quant_params);
        } else {
            paged_attn_memcpy(k, v, k_cache, v_cache, _slot_mapping);
        }
    }

    void execute(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr> outputs) override {
        PlainTensor q;
        PlainTensor k;
        PlainTensor v;
        PlainTensor k_cache;
        PlainTensor v_cache;
        PlainTensor past_lens;
        PlainTensor subsequence_begins;
        PlainTensor block_indices;
        PlainTensor block_indices_begins;
        float scale = NAN;
        size_t sliding_window = 0;
        PlainTensor alibi_slopes;
        size_t max_context_len = 0;
        PlainTensor score_aggregation_window;

        PlainTensor rotated_block_indices;
        PlainTensor rotation_deltas;
        PlainTensor rotation_trig_lut;

        PlainTensor xattention_threshold;
        int32_t xattention_block_size = 0;
        int32_t xattention_stride = 0;

        PlainTensor sinks;

        PlainTensor output_emb;
        PlainTensor output_score;

        init(inputs,
             outputs,
             q,
             k,
             v,
             k_cache,
             v_cache,
             past_lens,
             subsequence_begins,
             block_indices,
             block_indices_begins,
             scale,
             sliding_window,
             alibi_slopes,
             max_context_len,
             score_aggregation_window,
             rotated_block_indices,
             rotation_deltas,
             rotation_trig_lut,
             xattention_threshold,
             xattention_block_size,
             xattention_stride,
             sinks,
             output_emb,
             output_score);

        if (rotated_block_indices) {
            // Rotate kv cache currently doesn't support quantized cache.
            // for u8 it only supports compilation but throws exception in the runtime
            // TODO: implement u4/u8
            rotate_kv_cache<KEY_PREC>(k_cache,
                                      rotated_block_indices,
                                      rotation_deltas,
                                      rotation_trig_lut,
                                      _helper._block_rotation_coefficient_scratch);
        }

        concat_pastkv(k, v, k_cache, v_cache, past_lens, subsequence_begins, block_indices, block_indices_begins);

        _kernel(q,
                k_cache,
                v_cache,
                output_emb,
                output_score,
                max_context_len,
                past_lens,
                subsequence_begins,
                block_indices,
                block_indices_begins,
                alibi_slopes,
                score_aggregation_window);
    }
};
#endif

std::shared_ptr<PagedAttentionExecutor> make_pa_executor(ov::element::Type data_type,
                                                         ov::element::Type key_cache_type,
                                                         ov::element::Type value_cache_type,
                                                         const PagedAttnQuantParams& params) {
    std::shared_ptr<PagedAttentionExecutor> executor;
    if (params.is_sage_attn) {
        bool s8s8_available = (ov::with_cpu_x86_avx512_core_amx_int8() ||
                               dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::cpu_isa_t::avx2_vnni_2));
        OPENVINO_ASSERT(s8s8_available, "make_pa_executor: sage_attn needs amx_int8/vnni2 to support");
    }
#if defined(OPENVINO_ARCH_X86_64)
    if (data_type == ov::element::bf16) {
#    if defined(HAVE_AVX512F)
        if (key_cache_type == ov::element::i8 && params.is_sage_attn) {
            executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::element::i8, ov::element::u8>>(params);
        } else if (key_cache_type == ov::element::u8) {
            if (value_cache_type == ov::element::u4) {
                executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::element::u8, ov::element::u4>>(params);
            } else if (value_cache_type == ov::element::u8) {
                executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::element::u8, ov::element::u8>>(params);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u8 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }

        } else if (key_cache_type == ov::element::u4) {
            if (value_cache_type == ov::element::u4) {
                executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::element::u4, ov::element::u4>>(params);
            } else if (value_cache_type == ov::element::u8) {
                executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::element::u4, ov::element::u8>>(params);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u4 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }
        } else {
            OPENVINO_ASSERT(key_cache_type == ov::element::bf16 && value_cache_type == ov::element::bf16,
                            "expect kvcache type bf16, current: ",
                            key_cache_type,
                            " , ",
                            value_cache_type);
            executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::element::bf16, ov::element::bf16>>();
        }
#    else
        OPENVINO_THROW("make_pa_executor: bf16 needs avx512+ hardware.");
#    endif
    } else if (data_type == ov::element::f16) {
#    if defined(HAVE_AVX512F)
        if (key_cache_type == ov::element::i8 && params.is_sage_attn) {
            executor = std::make_shared<AttentionExecutor<ov::float16, ov::element::i8, ov::element::u8>>(params);
        } else if (key_cache_type == ov::element::u8) {
            if (value_cache_type == ov::element::u4) {
                executor = std::make_shared<AttentionExecutor<ov::float16, ov::element::u8, ov::element::u4>>(params);
            } else if (value_cache_type == ov::element::u8) {
                executor = std::make_shared<AttentionExecutor<ov::float16, ov::element::u8, ov::element::u8>>(params);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u8 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }
        } else if (key_cache_type == ov::element::u4) {
            if (value_cache_type == ov::element::u4) {
                executor = std::make_shared<AttentionExecutor<ov::float16, ov::element::u4, ov::element::u4>>(params);
            } else if (value_cache_type == ov::element::u8) {
                executor = std::make_shared<AttentionExecutor<ov::float16, ov::element::u4, ov::element::u8>>(params);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u4 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }
        } else {
            OPENVINO_ASSERT(key_cache_type == ov::element::f16 && value_cache_type == ov::element::f16,
                            "expect kvcache type f16, current: ",
                            key_cache_type,
                            " , ",
                            value_cache_type);
            executor = std::make_shared<AttentionExecutor<ov::float16, ov::element::f16, ov::element::f16>>();
        }
#    else
        OPENVINO_THROW("make_pa_executor: f16 needs avx512+ hardware.");
#    endif
    } else if (data_type == ov::element::f32) {
        if (key_cache_type == ov::element::i8 && params.is_sage_attn) {
            executor = std::make_shared<AttentionExecutor<float, ov::element::i8, ov::element::u8>>(params);
        } else if (key_cache_type == ov::element::u8) {
            if (value_cache_type == ov::element::u4) {
                executor = std::make_shared<AttentionExecutor<float, ov::element::u8, ov::element::u4>>(params);
            } else if (value_cache_type == ov::element::u8) {
                executor = std::make_shared<AttentionExecutor<float, ov::element::u8, ov::element::u8>>(params);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u8 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }
        } else if (key_cache_type == ov::element::u4) {
            if (value_cache_type == ov::element::u4) {
                executor = std::make_shared<AttentionExecutor<float, ov::element::u4, ov::element::u4>>(params);
            } else if (value_cache_type == ov::element::u8) {
                executor = std::make_shared<AttentionExecutor<float, ov::element::u4, ov::element::u8>>(params);
            } else {
                OPENVINO_THROW("make_pa_executor: key_cache_type u4 with value_cache_type ",
                               value_cache_type.to_string(),
                               " is not support");
            }
        } else if (key_cache_type == ov::element::f16) {
            OPENVINO_ASSERT(value_cache_type == ov::element::f16,
                            "expect value_cache_type type f16, current: ",
                            value_cache_type);
            executor = std::make_shared<AttentionExecutor<float, ov::element::f16, ov::element::f16>>(params);
        } else {
            OPENVINO_ASSERT(all_of(ov::element::f32, key_cache_type, value_cache_type),
                            "expect kvcache type f32, current: ",
                            key_cache_type,
                            " , ",
                            value_cache_type);
            executor = std::make_shared<AttentionExecutor<float, ov::element::f32, ov::element::f32>>(params);
        }
    } else {
        OPENVINO_THROW("make_pa_executor: unsupported precision: ", data_type);
    }
#elif (defined(OPENVINO_ARCH_ARM64) && defined(HAVE_SVE))
    if (data_type == ov::element::f32) {
        if (key_cache_type == ov::element::u8 && value_cache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<float, ov::element::u8, ov::element::u8>>(params);
        } else {
            OPENVINO_THROW("make_pa_executor: key_cache_type and value_cache_type of u8 is only support");
        }
    }
    if (data_type == ov::element::f16) {
        if (key_cache_type == ov::element::u8 && value_cache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<ov::float16, ov::element::u8, ov::element::u8>>(params);
        } else {
            OPENVINO_THROW("make_pa_executor: key_cache_type and value_cache_type of u8 is only support");
        }
    }

#else
    OPENVINO_THROW("make_pa_executor: only support x64 platform or ARM with SVE support");
#endif
    return executor;
}

}  // namespace ov::Extensions::Cpu::XARCH
