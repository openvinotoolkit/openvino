// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#define USE_FAST_EXP 0

#if USE_FAST_EXP
#include "fast_exp.h"
#else

#include "opt_exp.h"

#endif

#include <cmath>
#include "defs.h"
#include "ie_parallel.hpp"


static inline
void softmax_many_batches(const float *src_data, float *dst_data, int B, int C, int H, int W) {
    InferenceEngine::parallel_for(B * H * W, [&](size_t i) {
        const float *psrc = src_data + (i / (H * W)) * C * H * W - (i / (H * W)) * H * W;
        float *pdst = dst_data + (i / (H * W)) * C * H * W - (i / (H * W)) * H * W;

        float max = psrc[i];
        for (int c = 0; c < C; c++) {
            float val = psrc[c * H * W + i];
            if (val > max) max = val;
        }

        float expSum = 0;
        for (int c = 0; c < C; c++) {
            pdst[c * H * W + i] = exp(psrc[c * H * W + i] - max);
            expSum += pdst[c * H * W + i];
        }

        for (int c = 0; c < C; c++) {
            pdst[c * H * W + i] = pdst[c * H * W + i] / expSum;
        }
    });
}

static inline
void softmax_generic(const float *src_data, float *dst_data, int B, int C, int H, int W) {
    for (int b = 0; b < B; b++) {
#if defined(HAVE_AVX2)
        for (int i = 0; i <= H*W - 8; i += 8) {
            __m256 vmax = _mm256_loadu_ps(src_data + b*C*H*W + i);
            for (int c = 0; c < C; c++) {
                __m256 vval = _mm256_loadu_ps(src_data + b*C*H*W + c*H*W + i);
                __m256 vmask = _mm256_cmp_ps(vval, vmax, _CMP_GT_OS);
                vmax = _mm256_blendv_ps(vmax, vval, vmask);
            }

            __m256 vexpSum = _mm256_setzero_ps();
            for (int c = 0; c < C; c++) {
                __m256 vval = _mm256_loadu_ps(src_data + b*C*H*W + c*H*W + i);
#if USE_FAST_EXP
                __m256 vres = _avx_fast_exp_ps(_mm256_sub_ps(vval, vmax));
#else
                __m256 vres = _avx_opt_exp_ps(_mm256_sub_ps(vval, vmax));
#endif
                vexpSum = _mm256_add_ps(vexpSum, vres);
                _mm256_storeu_ps(dst_data + b*C*H*W + c*H*W + i, vres);
            }

            for (int c = 0; c < C; c++) {
                __m256 vval = _mm256_loadu_ps(dst_data + b*C*H*W + c*H*W + i);
                _mm256_storeu_ps(dst_data + b*C*H*W + c*H*W + i, _mm256_div_ps(vval, vexpSum));
            }
        }
#elif defined(HAVE_SSE)
        for (int i = 0; i <= H*W - 4; i += 4) {
            __m128 vmax = _mm_loadu_ps(src_data + b*C*H*W + i);
            for (int c = 0; c < C; c++) {
                __m128 vval = _mm_loadu_ps(src_data + b*C*H*W + c*H*W + i);
                __m128 vmask = _mm_cmpgt_ps(vval, vmax);
                vmax = _mm_blendv_ps(vmax, vval, vmask);
            }

            __m128 vexpSum = _mm_setzero_ps();
            for (int c = 0; c < C; c++) {
                __m128 vval = _mm_loadu_ps(src_data + b*C*H*W + c*H*W + i);
#if USE_FAST_EXP
                __m128 vres = _sse_fast_exp_ps(_mm_sub_ps(vval, vmax));
#else
                __m128 vres = _sse_opt_exp_ps(_mm_sub_ps(vval, vmax));
#endif
                vexpSum = _mm_add_ps(vexpSum, vres);
                _mm_storeu_ps(dst_data + b*C*H*W + c*H*W + i, vres);
            }

            for (int c = 0; c < C; c++) {
                __m128 vval = _mm_loadu_ps(dst_data + b*C*H*W + c*H*W + i);
                _mm_storeu_ps(dst_data + b*C*H*W + c*H*W + i, _mm_div_ps(vval, vexpSum));
            }
        }
#endif

#if defined(HAVE_AVX2)
        int start = (H*W / 8) * 8;
#elif defined(HAVE_SSE)
        int start = (H*W / 4) * 4;
#else
        int start = 0;
#endif
        for (int i = start; i < H * W; i++) {
            float max = src_data[b * C * H * W + i];
            for (int c = 0; c < C; c++) {
                float val = src_data[b * C * H * W + c * H * W + i];
                if (val > max) max = val;
            }

            float expSum = 0;
            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + i] = exp(src_data[b * C * H * W + c * H * W + i] - max);
                expSum += dst_data[b * C * H * W + c * H * W + i];
            }

            for (int c = 0; c < C; c++) {
                dst_data[b * C * H * W + c * H * W + i] = dst_data[b * C * H * W + c * H * W + i] / expSum;
            }
        }
    }
}