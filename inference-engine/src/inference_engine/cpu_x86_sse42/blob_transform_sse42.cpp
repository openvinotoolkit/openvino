// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "blob_transform_sse42.hpp"

#include <nmmintrin.h>  // SSE 4.2

namespace InferenceEngine {

//------------------------------------------------------------------------
//
// OpenCV universal intrinsics (refactored), x86 specific - require SSE4.1
//
//------------------------------------------------------------------------

static inline
void mm_load_deinterleave(const uint8_t* ptr, __m128i& a, __m128i& b, __m128i& c) {
    const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
    __m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + 16));
    __m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr + 32));
    __m128i a0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
    __m128i b0 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
    __m128i c0 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);
    const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
    const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
    const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    a0 = _mm_shuffle_epi8(a0, sh_b);
    b0 = _mm_shuffle_epi8(b0, sh_g);
    c0 = _mm_shuffle_epi8(c0, sh_r);
    a = a0;
    b = b0;
    c = c0;
}

static inline
void mm_load_deinterleave(const float* ptr, __m128& a, __m128& b, __m128& c) {
    __m128 t0 = _mm_loadu_ps(ptr + 0);
    __m128 t1 = _mm_loadu_ps(ptr + 4);
    __m128 t2 = _mm_loadu_ps(ptr + 8);

    __m128 at12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 1, 0, 2));
    a = _mm_shuffle_ps(t0, at12, _MM_SHUFFLE(2, 0, 3, 0));

    __m128 bt01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 bt12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 2, 0, 3));
    b = _mm_shuffle_ps(bt01, bt12, _MM_SHUFFLE(2, 0, 2, 0));

    __m128 ct01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 1, 0, 2));
    c = _mm_shuffle_ps(ct01, t2, _MM_SHUFFLE(3, 0, 2, 0));
}

static inline
void mm_store_interleave(uint8_t* ptr, __m128i a, __m128i b, __m128i c) {
    const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
    __m128i a0 = _mm_shuffle_epi8(a, sh_a);
    __m128i b0 = _mm_shuffle_epi8(b, sh_b);
    __m128i c0 = _mm_shuffle_epi8(c, sh_c);

    const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
    __m128i v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
    __m128i v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr)     , v0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + 16), v1);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(ptr + 32), v2);
}

static inline
void mm_store_interleave(float* ptr, __m128 a, __m128 b, __m128 c) {
    __m128 u0 = _mm_shuffle_ps(a , b , _MM_SHUFFLE(0, 0, 0, 0));
    __m128 u1 = _mm_shuffle_ps(c , a , _MM_SHUFFLE(1, 1, 0, 0));
    __m128 v0 = _mm_shuffle_ps(u0, u1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u2 = _mm_shuffle_ps(b , c , _MM_SHUFFLE(1, 1, 1, 1));
    __m128 u3 = _mm_shuffle_ps(a , b , _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1 = _mm_shuffle_ps(u2, u3, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u4 = _mm_shuffle_ps(c , a , _MM_SHUFFLE(3, 3, 2, 2));
    __m128 u5 = _mm_shuffle_ps(b , c , _MM_SHUFFLE(3, 3, 3, 3));
    __m128 v2 = _mm_shuffle_ps(u4, u5, _MM_SHUFFLE(2, 0, 2, 0));

    _mm_storeu_ps(ptr    , v0);
    _mm_storeu_ps(ptr + 4, v1);
    _mm_storeu_ps(ptr + 8, v2);
}

//------------------------------------------------------------------------
//
// Blob-copy primitives namually vectored for SSE 4.2 (w/o OpenMP threads)
//
//------------------------------------------------------------------------

void blob_copy_4d_split_u8c3(const uint8_t *src_ptr,
                                   uint8_t *dst_ptr,
                                    size_t  N_src_stride,
                                    size_t  H_src_stride,
                                    size_t  N_dst_stride,
                                    size_t  H_dst_stride,
                                    size_t  C_dst_stride,
                                       int  N,
                                       int  H,
                                       int  W) {
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++) {
        const uint8_t *src = src_ptr + n*N_src_stride + h*H_src_stride;
        uint8_t *dst0 = dst_ptr + n*N_dst_stride + 0*C_dst_stride + h*H_dst_stride;
        uint8_t *dst1 = dst_ptr + n*N_dst_stride + 1*C_dst_stride + h*H_dst_stride;
        uint8_t *dst2 = dst_ptr + n*N_dst_stride + 2*C_dst_stride + h*H_dst_stride;

        int w = 0;

        // SIMD128 manually
        for (; w < W - 16; w += 16) {
            __m128i r0, r1, r2;
            mm_load_deinterleave(&src[3 * w], r0, r1, r2);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst0 + w), r0);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst1 + w), r1);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst2 + w), r2);
        }

        for (; w < W; w++) {
            dst0[w] = src[3*w + 0];
            dst1[w] = src[3*w + 1];
            dst2[w] = src[3*w + 2];
        }
    }
}

void blob_copy_4d_split_f32c3(const float *src_ptr,
                                    float *dst_ptr,
                                   size_t  N_src_stride,
                                   size_t  H_src_stride,
                                   size_t  N_dst_stride,
                                   size_t  H_dst_stride,
                                   size_t  C_dst_stride,
                                      int  N,
                                      int  H,
                                      int  W) {
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++) {
        const float *src = src_ptr + n*N_src_stride + h*H_src_stride;
        float *dst0 = dst_ptr + n*N_dst_stride + 0*C_dst_stride + h*H_dst_stride;
        float *dst1 = dst_ptr + n*N_dst_stride + 1*C_dst_stride + h*H_dst_stride;
        float *dst2 = dst_ptr + n*N_dst_stride + 2*C_dst_stride + h*H_dst_stride;

        int w = 0;

        // SIMD128 manually
        for (; w < W - 4; w += 4) {
            __m128 r0, r1, r2;
            mm_load_deinterleave(&src[3 * w], r0, r1, r2);
            _mm_storeu_ps(&dst0[w], r0);
            _mm_storeu_ps(&dst1[w], r1);
            _mm_storeu_ps(&dst2[w], r2);
        }

        for (; w < W; w++) {
            dst0[w] = src[3*w + 0];
            dst1[w] = src[3*w + 1];
            dst2[w] = src[3*w + 2];
        }
    }
}

void blob_copy_4d_merge_u8c3(const uint8_t *src_ptr,
                                   uint8_t *dst_ptr,
                                    size_t  N_src_stride,
                                    size_t  H_src_stride,
                                    size_t  C_src_stride,
                                    size_t  N_dst_stride,
                                    size_t  H_dst_stride,
                                       int  N,
                                       int  H,
                                       int  W) {
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++) {
        const uint8_t *src0 = src_ptr + n*N_src_stride + 0*C_src_stride + h*H_src_stride;
        const uint8_t *src1 = src_ptr + n*N_src_stride + 1*C_src_stride + h*H_src_stride;
        const uint8_t *src2 = src_ptr + n*N_src_stride + 2*C_src_stride + h*H_src_stride;

        uint8_t *dst = dst_ptr + n*N_dst_stride + h*H_dst_stride;

        int w = 0;

        // SIMD128 manually
        for (; w < W - 16; w += 16) {
            __m128i r0, r1, r2;
            r0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + w));
            r1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + w));
            r2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2 + w));
            mm_store_interleave(&dst[3 * w], r0, r1, r2);
        }

        for (; w < W; w++) {
            dst[3*w + 0] = src0[w];
            dst[3*w + 1] = src1[w];
            dst[3*w + 2] = src2[w];
        }
    }
}

void blob_copy_4d_merge_f32c3(const float *src_ptr,
                                    float *dst_ptr,
                                   size_t  N_src_stride,
                                   size_t  H_src_stride,
                                   size_t  C_src_stride,
                                   size_t  N_dst_stride,
                                   size_t  H_dst_stride,
                                      int  N,
                                      int  H,
                                      int  W) {
    for (int n = 0; n < N; n++)
    for (int h = 0; h < H; h++) {
        const float *src0 = src_ptr + n*N_src_stride + 0*C_src_stride + h*H_src_stride;
        const float *src1 = src_ptr + n*N_src_stride + 1*C_src_stride + h*H_src_stride;
        const float *src2 = src_ptr + n*N_src_stride + 2*C_src_stride + h*H_src_stride;

        float *dst = dst_ptr + n*N_dst_stride + h*H_dst_stride;

        int w = 0;

        // SIMD128 manually
        for (; w < W - 4; w += 4) {
            __m128 r0, r1, r2;
            r0 = _mm_loadu_ps(&src0[w]);
            r1 = _mm_loadu_ps(&src1[w]);
            r2 = _mm_loadu_ps(&src2[w]);
            mm_store_interleave(&dst[3 * w], r0, r1, r2);
        }

        for (; w < W; w++) {
            dst[3*w + 0] = src0[w];
            dst[3*w + 1] = src1[w];
            dst[3*w + 2] = src2[w];
        }
    }
}

}  // namespace InferenceEngine
