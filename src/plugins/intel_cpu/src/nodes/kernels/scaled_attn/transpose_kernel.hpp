// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common.hpp"
#include "openvino/core/type/element_type.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

#if defined(HAVE_AVX512F)
inline void transpose_m512i_16x16(__m512i& r0, __m512i& r1, __m512i& r2, __m512i& r3,
    __m512i& r4, __m512i& r5, __m512i& r6, __m512i& r7,
    __m512i& r8, __m512i& r9, __m512i& ra, __m512i& rb,
    __m512i& rc, __m512i& rd, __m512i& re, __m512i& rf) {
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;

    t0 = _mm512_unpacklo_epi32(r0, r1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
    t1 = _mm512_unpackhi_epi32(r0, r1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    t2 = _mm512_unpacklo_epi32(r2, r3); //  32  48  33  49 ...
    t3 = _mm512_unpackhi_epi32(r2, r3); //  34  50  35  51 ...
    t4 = _mm512_unpacklo_epi32(r4, r5); //  64  80  65  81 ...
    t5 = _mm512_unpackhi_epi32(r4, r5); //  66  82  67  83 ...
    t6 = _mm512_unpacklo_epi32(r6, r7); //  96 112  97 113 ...
    t7 = _mm512_unpackhi_epi32(r6, r7); //  98 114  99 115 ...
    t8 = _mm512_unpacklo_epi32(r8, r9); // 128 ...
    t9 = _mm512_unpackhi_epi32(r8, r9); // 130 ...
    ta = _mm512_unpacklo_epi32(ra, rb); // 160 ...
    tb = _mm512_unpackhi_epi32(ra, rb); // 162 ...
    tc = _mm512_unpacklo_epi32(rc, rd); // 196 ...
    td = _mm512_unpackhi_epi32(rc, rd); // 198 ...
    te = _mm512_unpacklo_epi32(re, rf); // 228 ...
    tf = _mm512_unpackhi_epi32(re, rf); // 230 ...

    r0 = _mm512_unpacklo_epi64(t0, t2); //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(t0, t2); //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(t1, t3); //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(t1, t3); //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(t4, t6); //  64  80  96 112 ...
    r5 = _mm512_unpackhi_epi64(t4, t6); //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(t5, t7); //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(t5, t7); //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(t8, ta); // 128 144 160 176 ...
    r9 = _mm512_unpackhi_epi64(t8, ta); // 129 145 161 178 ...
    ra = _mm512_unpacklo_epi64(t9, tb); // 130 146 162 177 ...
    rb = _mm512_unpackhi_epi64(t9, tb); // 131 147 163 179 ...
    rc = _mm512_unpacklo_epi64(tc, te); // 192 208 228 240 ...
    rd = _mm512_unpackhi_epi64(tc, te); // 193 209 229 241 ...
    re = _mm512_unpacklo_epi64(td, tf); // 194 210 230 242 ...
    rf = _mm512_unpackhi_epi64(td, tf); // 195 211 231 243 ...

    t0 = _mm512_shuffle_i32x4(r0, r4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
    t1 = _mm512_shuffle_i32x4(r1, r5, 0x88); //   1  17  33  49 ...
    t2 = _mm512_shuffle_i32x4(r2, r6, 0x88); //   2  18  34  50 ...
    t3 = _mm512_shuffle_i32x4(r3, r7, 0x88); //   3  19  35  51 ...
    t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd); //   4  20  36  52 ...
    t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd); //   5  21  37  53 ...
    t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd); //   6  22  38  54 ...
    t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd); //   7  23  39  55 ...
    t8 = _mm512_shuffle_i32x4(r8, rc, 0x88); // 128 144 160 176 ...
    t9 = _mm512_shuffle_i32x4(r9, rd, 0x88); // 129 145 161 177 ...
    ta = _mm512_shuffle_i32x4(ra, re, 0x88); // 130 146 162 178 ...
    tb = _mm512_shuffle_i32x4(rb, rf, 0x88); // 131 147 163 179 ...
    tc = _mm512_shuffle_i32x4(r8, rc, 0xdd); // 132 148 164 180 ...
    td = _mm512_shuffle_i32x4(r9, rd, 0xdd); // 133 149 165 181 ...
    te = _mm512_shuffle_i32x4(ra, re, 0xdd); // 134 150 166 182 ...
    tf = _mm512_shuffle_i32x4(rb, rf, 0xdd); // 135 151 167 183 ...

    r0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
    r1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
    r2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
    r3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
    r4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
    r5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
    r6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
    r7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
    r8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
    r9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
    ra = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
    rb = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
    rc = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
    rd = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
    re = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
    rf = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
}

template<typename T>
inline void transpose_16x16_kernel(float* _dst, T* src, size_t dst_stride, size_t src_stride) {
    auto* dst = reinterpret_cast<uint32_t*>(_dst);
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    r0 = _mm512_castps_si512(mm512_uni_loadu_ps(src));
    r1 = _mm512_castps_si512(mm512_uni_loadu_ps(src + src_stride));
    r2 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 2 * src_stride));
    r3 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 3 * src_stride));
    r4 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 4 * src_stride));
    r5 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 5 * src_stride));
    r6 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 6 * src_stride));
    r7 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 7 * src_stride));
    r8 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 8 * src_stride));
    r9 = _mm512_castps_si512(mm512_uni_loadu_ps(src + 9 * src_stride));
    ra = _mm512_castps_si512(mm512_uni_loadu_ps(src + 10 * src_stride));
    rb = _mm512_castps_si512(mm512_uni_loadu_ps(src + 11 * src_stride));
    rc = _mm512_castps_si512(mm512_uni_loadu_ps(src + 12 * src_stride));
    rd = _mm512_castps_si512(mm512_uni_loadu_ps(src + 13 * src_stride));
    re = _mm512_castps_si512(mm512_uni_loadu_ps(src + 14 * src_stride));
    rf = _mm512_castps_si512(mm512_uni_loadu_ps(src + 15 * src_stride));

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

    _mm512_storeu_si512(dst, r0);
    _mm512_storeu_si512(dst + dst_stride, r1);
    _mm512_storeu_si512(dst + 2 * dst_stride, r2);
    _mm512_storeu_si512(dst + 3 * dst_stride, r3);
    _mm512_storeu_si512(dst + 4 * dst_stride, r4);
    _mm512_storeu_si512(dst + 5 * dst_stride, r5);
    _mm512_storeu_si512(dst + 6 * dst_stride, r6);
    _mm512_storeu_si512(dst + 7 * dst_stride, r7);
    _mm512_storeu_si512(dst + 8 * dst_stride, r8);
    _mm512_storeu_si512(dst + 9 * dst_stride, r9);
    _mm512_storeu_si512(dst + 10 * dst_stride, ra);
    _mm512_storeu_si512(dst + 11 * dst_stride, rb);
    _mm512_storeu_si512(dst + 12 * dst_stride, rc);
    _mm512_storeu_si512(dst + 13 * dst_stride, rd);
    _mm512_storeu_si512(dst + 14 * dst_stride, re);
    _mm512_storeu_si512(dst + 15 * dst_stride, rf);
}

template<typename T>
inline void transpose_16xK_kernel(float* _dst, T* src, size_t K, size_t dst_stride, size_t src_stride) {
    auto* dst = reinterpret_cast<uint32_t*>(_dst);
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    r0 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src, K));
    r1 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + src_stride, K));
    r2 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 2 * src_stride, K));
    r3 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 3 * src_stride, K));
    r4 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 4 * src_stride, K));
    r5 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 5 * src_stride, K));
    r6 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 6 * src_stride, K));
    r7 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 7 * src_stride, K));
    r8 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 8 * src_stride, K));
    r9 = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 9 * src_stride, K));
    ra = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 10 * src_stride, K));
    rb = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 11 * src_stride, K));
    rc = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 12 * src_stride, K));
    rd = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 13 * src_stride, K));
    re = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 14 * src_stride, K));
    rf = _mm512_castps_si512(mm512_uni_loadu_tail_ps(src + 15 * src_stride, K));

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

#define S(m) _mm512_storeu_si512(dst + 0x##m * dst_stride, r##m)
#define S8() S(0); S(1); S(2); S(3); S(4); S(5); S(6); S(7);
    switch (K) {
        case 8: S8(); break;
        case 9: S8() S(8); break;
        case 10: S8(); S(8); S(9); break;
        case 11: S8(); S(8); S(9); S(a); break;
        case 12: S8(); S(8); S(9); S(a); S(b); break;
        case 13: S8(); S(8); S(9); S(a); S(b); S(c); break;
        case 14: S8(); S(8); S(9); S(a); S(b); S(c); S(d); break;
        case 15: S8(); S(8); S(9); S(a); S(b); S(c); S(d); S(e); break;
        case 1: S(0); break;
        case 2: S(0); S(1); break;
        case 3: S(0); S(1); S(2); break;
        case 4: S(0); S(1); S(2); S(3); break;
        case 5: S(0); S(1); S(2); S(3); S(4); break;
        case 6: S(0); S(1); S(2); S(3); S(4); S(5); break;
        case 7: S(0); S(1); S(2); S(3); S(4); S(5); S(6); break;
    }
}

inline void transpose_16x16_kernel(uint32_t* dst, uint32_t* src, size_t dst_stride, size_t src_stride) {
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    r0 = _mm512_loadu_si512(src);
    r1 = _mm512_loadu_si512(src + src_stride);
    r2 = _mm512_loadu_si512(src + 2 * src_stride);
    r3 = _mm512_loadu_si512(src + 3 * src_stride);
    r4 = _mm512_loadu_si512(src + 4 * src_stride);
    r5 = _mm512_loadu_si512(src + 5 * src_stride);
    r6 = _mm512_loadu_si512(src + 6 * src_stride);
    r7 = _mm512_loadu_si512(src + 7 * src_stride);
    r8 = _mm512_loadu_si512(src + 8 * src_stride);
    r9 = _mm512_loadu_si512(src + 9 * src_stride);
    ra = _mm512_loadu_si512(src + 10 * src_stride);
    rb = _mm512_loadu_si512(src + 11 * src_stride);
    rc = _mm512_loadu_si512(src + 12 * src_stride);
    rd = _mm512_loadu_si512(src + 13 * src_stride);
    re = _mm512_loadu_si512(src + 14 * src_stride);
    rf = _mm512_loadu_si512(src + 15 * src_stride);

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

    _mm512_storeu_si512(dst, r0);
    _mm512_storeu_si512(dst + dst_stride, r1);
    _mm512_storeu_si512(dst + 2 * dst_stride, r2);
    _mm512_storeu_si512(dst + 3 * dst_stride, r3);
    _mm512_storeu_si512(dst + 4 * dst_stride, r4);
    _mm512_storeu_si512(dst + 5 * dst_stride, r5);
    _mm512_storeu_si512(dst + 6 * dst_stride, r6);
    _mm512_storeu_si512(dst + 7 * dst_stride, r7);
    _mm512_storeu_si512(dst + 8 * dst_stride, r8);
    _mm512_storeu_si512(dst + 9 * dst_stride, r9);
    _mm512_storeu_si512(dst + 10 * dst_stride, ra);
    _mm512_storeu_si512(dst + 11 * dst_stride, rb);
    _mm512_storeu_si512(dst + 12 * dst_stride, rc);
    _mm512_storeu_si512(dst + 13 * dst_stride, rd);
    _mm512_storeu_si512(dst + 14 * dst_stride, re);
    _mm512_storeu_si512(dst + 15 * dst_stride, rf);
}

inline void transpose_16xK_kernel(uint32_t* dst, uint32_t* src, size_t K, size_t dst_stride, size_t src_stride) {
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    __mmask16 k = 0xffff >> (16 - K);

    r0 = _mm512_maskz_loadu_epi32(k, src);
    r1 = _mm512_maskz_loadu_epi32(k, src + src_stride);
    r2 = _mm512_maskz_loadu_epi32(k, src + 2 * src_stride);
    r3 = _mm512_maskz_loadu_epi32(k, src + 3 * src_stride);
    r4 = _mm512_maskz_loadu_epi32(k, src + 4 * src_stride);
    r5 = _mm512_maskz_loadu_epi32(k, src + 5 * src_stride);
    r6 = _mm512_maskz_loadu_epi32(k, src + 6 * src_stride);
    r7 = _mm512_maskz_loadu_epi32(k, src + 7 * src_stride);
    r8 = _mm512_maskz_loadu_epi32(k, src + 8 * src_stride);
    r9 = _mm512_maskz_loadu_epi32(k, src + 9 * src_stride);
    ra = _mm512_maskz_loadu_epi32(k, src + 10 * src_stride);
    rb = _mm512_maskz_loadu_epi32(k, src + 11 * src_stride);
    rc = _mm512_maskz_loadu_epi32(k, src + 12 * src_stride);
    rd = _mm512_maskz_loadu_epi32(k, src + 13 * src_stride);
    re = _mm512_maskz_loadu_epi32(k, src + 14 * src_stride);
    rf = _mm512_maskz_loadu_epi32(k, src + 15 * src_stride);

    transpose_m512i_16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf);

    switch (K) {
        case 8: S8(); break;
        case 9: S8() S(8); break;
        case 10: S8(); S(8); S(9); break;
        case 11: S8(); S(8); S(9); S(a); break;
        case 12: S8(); S(8); S(9); S(a); S(b); break;
        case 13: S8(); S(8); S(9); S(a); S(b); S(c); break;
        case 14: S8(); S(8); S(9); S(a); S(b); S(c); S(d); break;
        case 15: S8(); S(8); S(9); S(a); S(b); S(c); S(d); S(e); break;
        case 1: S(0); break;
        case 2: S(0); S(1); break;
        case 3: S(0); S(1); S(2); break;
        case 4: S(0); S(1); S(2); S(3); break;
        case 5: S(0); S(1); S(2); S(3); S(4); break;
        case 6: S(0); S(1); S(2); S(3); S(4); S(5); break;
        case 7: S(0); S(1); S(2); S(3); S(4); S(5); S(6); break;
    }
#undef S
#undef S8
}

#elif defined(HAVE_AVX2)

// https://stackoverflow.com/questions/25622745/transpose-an-8x8-float-using-avx-avx2
inline void transpose_8x8(__m256& r0, __m256& r1, __m256& r2, __m256& r3, __m256& r4, __m256& r5, __m256& r6, __m256& r7) {
    __m256 t0, t1, t2, t3, t4, t5, t6, t7;
    __m256 tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7;
    t0 = _mm256_unpacklo_ps(r0, r1);
    t1 = _mm256_unpackhi_ps(r0, r1);
    t2 = _mm256_unpacklo_ps(r2, r3);
    t3 = _mm256_unpackhi_ps(r2, r3);
    t4 = _mm256_unpacklo_ps(r4, r5);
    t5 = _mm256_unpackhi_ps(r4, r5);
    t6 = _mm256_unpacklo_ps(r6, r7);
    t7 = _mm256_unpackhi_ps(r6, r7);
    tt0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    tt1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    tt2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    tt3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    tt4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    tt5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    tt6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    tt7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    r0 = _mm256_permute2f128_ps(tt0, tt4, 0x20);
    r1 = _mm256_permute2f128_ps(tt1, tt5, 0x20);
    r2 = _mm256_permute2f128_ps(tt2, tt6, 0x20);
    r3 = _mm256_permute2f128_ps(tt3, tt7, 0x20);
    r4 = _mm256_permute2f128_ps(tt0, tt4, 0x31);
    r5 = _mm256_permute2f128_ps(tt1, tt5, 0x31);
    r6 = _mm256_permute2f128_ps(tt2, tt6, 0x31);
    r7 = _mm256_permute2f128_ps(tt3, tt7, 0x31);
}

template<typename T>
inline void transpose_16x16_kernel(float* dst, T* src, size_t dst_stride, size_t src_stride) {
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;

    for (int i = 0; i < 16; i += 8) {
        for (int j = 0; j < 16; j += 8) {
            r0 = mm256_uni_loadu_ps(src + src_stride * j);
            r1 = mm256_uni_loadu_ps(src + src_stride * (1 + j));
            r2 = mm256_uni_loadu_ps(src + src_stride * (2 + j));
            r3 = mm256_uni_loadu_ps(src + src_stride * (3 + j));
            r4 = mm256_uni_loadu_ps(src + src_stride * (4 + j));
            r5 = mm256_uni_loadu_ps(src + src_stride * (5 + j));
            r6 = mm256_uni_loadu_ps(src + src_stride * (6 + j));
            r7 = mm256_uni_loadu_ps(src + src_stride * (7 + j));

            transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

            _mm256_storeu_ps(dst + j, r0);
            _mm256_storeu_ps(dst + j + dst_stride, r1);
            _mm256_storeu_ps(dst + j + dst_stride * 2, r2);
            _mm256_storeu_ps(dst + j + dst_stride * 3, r3);
            _mm256_storeu_ps(dst + j + dst_stride * 4, r4);
            _mm256_storeu_ps(dst + j + dst_stride * 5, r5);
            _mm256_storeu_ps(dst + j + dst_stride * 6, r6);
            _mm256_storeu_ps(dst + j + dst_stride * 7, r7);
        }
        src += 8;
        dst += 8 * dst_stride;
    }
}

template<typename T>
inline void transpose_16xK_kernel(float* dst, T* src, size_t K, size_t dst_stride, size_t src_stride) {
    __m256 r0, r1, r2, r3, r4, r5, r6, r7;

    if (K >= 8) {
        for (int j = 0; j < 16; j += 8) {
            r0 = mm256_uni_loadu_ps(src + src_stride * j);
            r1 = mm256_uni_loadu_ps(src + src_stride * (1 + j));
            r2 = mm256_uni_loadu_ps(src + src_stride * (2 + j));
            r3 = mm256_uni_loadu_ps(src + src_stride * (3 + j));
            r4 = mm256_uni_loadu_ps(src + src_stride * (4 + j));
            r5 = mm256_uni_loadu_ps(src + src_stride * (5 + j));
            r6 = mm256_uni_loadu_ps(src + src_stride * (6 + j));
            r7 = mm256_uni_loadu_ps(src + src_stride * (7 + j));

            transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

            _mm256_storeu_ps(dst + j, r0);
            _mm256_storeu_ps(dst + j + dst_stride, r1);
            _mm256_storeu_ps(dst + j + dst_stride * 2, r2);
            _mm256_storeu_ps(dst + j + dst_stride * 3, r3);
            _mm256_storeu_ps(dst + j + dst_stride * 4, r4);
            _mm256_storeu_ps(dst + j + dst_stride * 5, r5);
            _mm256_storeu_ps(dst + j + dst_stride * 6, r6);
            _mm256_storeu_ps(dst + j + dst_stride * 7, r7);
        }
        src += 8;
        dst += 8 * dst_stride;
        K -= 8;
    }
    if (K > 0) {
        for (int j = 0; j < 16; j += 8) {
            r0 = mm256_uni_loadu_tail_ps(src + src_stride * j, K);
            r1 = mm256_uni_loadu_tail_ps(src + src_stride * (1 + j), K);
            r2 = mm256_uni_loadu_tail_ps(src + src_stride * (2 + j), K);
            r3 = mm256_uni_loadu_tail_ps(src + src_stride * (3 + j), K);
            r4 = mm256_uni_loadu_tail_ps(src + src_stride * (4 + j), K);
            r5 = mm256_uni_loadu_tail_ps(src + src_stride * (5 + j), K);
            r6 = mm256_uni_loadu_tail_ps(src + src_stride * (6 + j), K);
            r7 = mm256_uni_loadu_tail_ps(src + src_stride * (7 + j), K);

            transpose_8x8(r0, r1, r2, r3, r4, r5, r6, r7);

#define S(m) _mm256_storeu_ps(dst + j + m * dst_stride, r##m)
            switch (K) {
                case 1: S(0); break;
                case 2: S(0); S(1); break;
                case 3: S(0); S(1); S(2); break;
                case 4: S(0); S(1); S(2); S(3); break;
                case 5: S(0); S(1); S(2); S(3); S(4); break;
                case 6: S(0); S(1); S(2); S(3); S(4); S(5); break;
                case 7: S(0); S(1); S(2); S(3); S(4); S(5); S(6); break;
            }
#undef S
        }
    }
}

#else

template<typename TSRC, typename TDST>
inline void transpose_16x16_kernel(TDST* dst, TSRC* src, size_t dst_stride, size_t src_stride) {
    for (size_t i = 0; i < 16; i++) {
        for (size_t j = 0; j < 16; j++) {
            dst[i * dst_stride + j] = static_cast<TDST>(src[i + j * src_stride]);
        }
    }
}

template<typename TSRC, typename TDST>
inline void transpose_16xK_kernel(TDST* dst, TSRC* src, size_t K, size_t dst_stride, size_t src_stride) {
    for (size_t i = 0; i < K; i++) {
        for (size_t j = 0; j < 16; j++) {
            dst[i * dst_stride + j] = static_cast<TDST>(src[i + j * src_stride]);
        }
    }
}

#endif

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov