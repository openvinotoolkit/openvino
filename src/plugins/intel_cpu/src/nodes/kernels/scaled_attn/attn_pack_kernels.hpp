// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// Packing of 16-bit key / value rows (bf16 / f16) into the AMX B operand layout (AVX-512 builds only): row-major VNNI
// B[k / 2][ldb][2]. pack_32NxK reads a contiguous matrix (PagedAttention's value pages), the *_block variants read
// every row through its own pointer (rows gathered from several places, as the sparse attention does).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/transpose_kernel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/general_utils.h"
#include "utils/plain_tensor.hpp"

#if defined(OPENVINO_ARCH_X86_64) && defined(HAVE_AVX512F)

namespace ov::Extensions::Cpu::XARCH {

template <
    typename T,
    typename = typename std::enable_if_t<(std::is_same_v<T, ov::bfloat16> || std::is_same_v<T, ov::float16>), bool>>
inline void pack_32x32_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
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
inline void pack_32x16_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
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
inline void pack_32xK_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride, size_t K) {
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
          typename std::enable_if_t<ov::intel_cpu::precision_of<TDST>::value != ov::element::f32 &&
                                        ov::intel_cpu::any_of(SRC_PREC, ov::element::bf16, ov::element::f16),
                                    bool> = true>
inline void pack_32NxK(TDST* dst,
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

// B operand packing straight from the cache rows: each row is read through its own pointer (copy_buffer_b needs one
// contiguous matrix) and written in the layout of copy_buffer_b and of PagedAttention's pack_32NxK / transpose_16NxK:
// 16-bit row-major VNNI B[k / 2][ldb][2], ldb = N rounded up to 32.

/// @brief Whether the packing below can be used: 16-bit rows of a multiple of 32 elements (the widths of the vector
/// loads). Other head sizes gather the rows and use copy_buffer_b.
[[nodiscard]] inline bool fused_pack_supported(size_t k_head_size, size_t v_head_size) {
    return k_head_size % 32 == 0 && v_head_size % 32 == 0;
}

/// @brief B operand of the QK product (keys, b_transposed, N = 32): dst[d][n] = the 32-bit pair d of the key row n, so
/// a 32 x (k_head_size / 2) transpose of 32-bit values, 16 x 16 at a time.
template <typename T>
void pack_k_block(const T* const* kv_ptrs, size_t k_head_size, T* dst) {
    static_assert(sizeof(T) == 2, "the rows are packed as 16-bit data (bf16 / f16)");
    auto* out = reinterpret_cast<uint32_t*>(dst);
    const size_t pairs = k_head_size / 2;
    for (size_t half = 0; half < 2; half++) {
        const auto* const* r = reinterpret_cast<const uint32_t* const*>(kv_ptrs + half * 16);
        for (size_t d = 0; d < pairs; d += 16) {
            __m512i r0 = _mm512_loadu_si512(r[0] + d);
            __m512i r1 = _mm512_loadu_si512(r[1] + d);
            __m512i r2 = _mm512_loadu_si512(r[2] + d);
            __m512i r3 = _mm512_loadu_si512(r[3] + d);
            __m512i r4 = _mm512_loadu_si512(r[4] + d);
            __m512i r5 = _mm512_loadu_si512(r[5] + d);
            __m512i r6 = _mm512_loadu_si512(r[6] + d);
            __m512i r7 = _mm512_loadu_si512(r[7] + d);
            __m512i r8 = _mm512_loadu_si512(r[8] + d);
            __m512i r9 = _mm512_loadu_si512(r[9] + d);
            __m512i ra = _mm512_loadu_si512(r[10] + d);
            __m512i rb = _mm512_loadu_si512(r[11] + d);
            __m512i rc = _mm512_loadu_si512(r[12] + d);
            __m512i rd = _mm512_loadu_si512(r[13] + d);
            __m512i re = _mm512_loadu_si512(r[14] + d);
            __m512i rf = _mm512_loadu_si512(r[15] + d);
            transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);
            uint32_t* o = out + d * 32 + half * 16;
            _mm512_storeu_si512(o, r0);
            _mm512_storeu_si512(o + 32, r1);
            _mm512_storeu_si512(o + 2 * 32, r2);
            _mm512_storeu_si512(o + 3 * 32, r3);
            _mm512_storeu_si512(o + 4 * 32, r4);
            _mm512_storeu_si512(o + 5 * 32, r5);
            _mm512_storeu_si512(o + 6 * 32, r6);
            _mm512_storeu_si512(o + 7 * 32, r7);
            _mm512_storeu_si512(o + 8 * 32, r8);
            _mm512_storeu_si512(o + 9 * 32, r9);
            _mm512_storeu_si512(o + 10 * 32, ra);
            _mm512_storeu_si512(o + 11 * 32, rb);
            _mm512_storeu_si512(o + 12 * 32, rc);
            _mm512_storeu_si512(o + 13 * 32, rd);
            _mm512_storeu_si512(o + 14 * 32, re);
            _mm512_storeu_si512(o + 15 * 32, rf);
        }
    }
}

/// @brief B operand of the P.V product (values, K = 32 rows): the row pair (2i, 2i + 1) interleaved element by element
/// into the row i of dst, whose stride is 2 * v_head_size elements.
template <typename T>
void pack_v_block(const T* const* kv_ptrs, size_t v_head_size, T* dst) {
    static_assert(sizeof(T) == 2, "the rows are packed as 16-bit data (bf16 / f16)");
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const auto midx = _mm512_loadu_si512(idx);
    auto* out_row = reinterpret_cast<uint16_t*>(dst);
    for (size_t i = 0; i < 16; i++) {
        const auto* row_a = reinterpret_cast<const uint16_t*>(kv_ptrs[2 * i]);
        const auto* row_b = reinterpret_cast<const uint16_t*>(kv_ptrs[2 * i + 1]);
        for (size_t k = 0; k < v_head_size; k += 32) {
            const auto a = _mm512_permutexvar_epi64(midx, _mm512_loadu_si512(row_a + k));
            const auto b = _mm512_permutexvar_epi64(midx, _mm512_loadu_si512(row_b + k));
            _mm512_storeu_si512(out_row + 2 * k, _mm512_unpacklo_epi16(a, b));
            _mm512_storeu_si512(out_row + 2 * k + 32, _mm512_unpackhi_epi16(a, b));
        }
        out_row += 2 * v_head_size;
    }
}

}  // namespace ov::Extensions::Cpu::XARCH

#endif  // OPENVINO_ARCH_X86_64 && HAVE_AVX512F
