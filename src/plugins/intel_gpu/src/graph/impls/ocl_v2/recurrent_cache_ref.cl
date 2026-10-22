// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// These modes fuse llama.cpp Qwen3.5 cache_r/cache_s maintenance. Runtime maps preserve
// slot reset/reorder and ordered MTP snapshot writes while source-cache reads stay immutable.
#if HISTORY < 0
KERNEL(recurrent_cache_ref)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* cache,
 const __global INPUT1_TYPE* states,
 const __global INPUT2_TYPE* control,
#if FUSED_PREPARE
 const __global INPUT3_TYPE* extra_rows,
 const __global INPUT4_TYPE* reset_begin,
 const __global INPUT5_TYPE* reset_len,
 const __global INPUT6_TYPE* extra_offsets,
#endif
 __global OUTPUT_TYPE* updated) {
    const size_t i = get_global_id(0);
    const int F = INPUT0_SIZE_X;
    if (i >= (size_t)INPUT0_SIZE_Y * F)
        return;
    const int dst = i / F, f = i % F;
    const int begin = control[INPUT2_GET_INDEX(0, 0, 0, 0)];
    const int B = control[INPUT2_GET_INDEX(0, 0, 0, 1)];
    const int stride = control[INPUT2_GET_INDEX(0, 0, 0, 2)];
    const int src_begin = control[INPUT2_GET_INDEX(0, 0, 0, 3)];
    const int N = INPUT1_SIZE_Y - src_begin;
#    if STATE_KEY_SIZE > 0
    const int matrix = STATE_KEY_SIZE * STATE_VALUE_SIZE;
    const int sf = (f / matrix) * matrix + (f % STATE_KEY_SIZE) * STATE_VALUE_SIZE + (f / STATE_KEY_SIZE) % STATE_VALUE_SIZE;
#    else
    const int sf = f;
#    endif
    int row = dst;
#if FUSED_PREPARE
    const int rb = reset_begin[INPUT4_OFFSET], re = rb + reset_len[INPUT5_OFFSET];
    const int eb = extra_offsets[INPUT6_GET_INDEX(0, 0, 0, 0)];
    const int es = extra_offsets[INPUT6_GET_INDEX(0, 0, 0, 1)];
    const int E = INPUT3_SIZE_X - es;
    if (dst >= eb && dst < eb + E)
        row = extra_rows[INPUT3_GET_INDEX(0, 0, 0, dst - eb + es)];
    float value = row >= rb && row < re ? 0.0f : convert_float(cache[INPUT0_GET_INDEX(0, 0, row, f)]);
#else
    float value = convert_float(cache[INPUT0_GET_INDEX(0, 0, row, f)]);
#endif
    // Later source planes have ordered-write precedence, including overlaps.
    for (int plane = 0; plane * B < N; ++plane) {
        const int b = dst - begin - plane * stride;
        if (b >= 0 && b < B && plane * B + b < N)
            value = convert_float(states[INPUT1_GET_INDEX(0, 0, src_begin + plane * B + b, sf)]);
    }
    updated[OUTPUT_GET_INDEX(0, 0, dst, f)] = TO_OUTPUT_TYPE(value);
}
#else
// Input/output allocations are distinct. Each element has one writer; source cycles and
// repeated gather indices are safe, including reset-before-gather of a source row.
KERNEL(recurrent_cache_ref)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* old_cache,
 const __global INPUT1_TYPE* active_rows,
 const __global INPUT2_TYPE* extra_rows,
 const __global INPUT3_TYPE* reset_begin,
 const __global INPUT4_TYPE* reset_len,
 const __global INPUT5_TYPE* extra_begin,
#    if HISTORY > 0
 const __global INPUT6_TYPE* qkv,
 const __global INPUT7_TYPE* windows,
#        if FUSED_CONV
 const __global INPUT8_TYPE* weights,
#        endif
#    endif
 __global OUTPUT_TYPE* active
#if PREPARE_CACHE
 , __global OUTPUT1_TYPE* updated
#endif
 ) {
    const size_t i = get_global_id(0);
    const int F = INPUT0_SIZE_X;
    const int S = INPUT0_SIZE_Y;
    const int B = INPUT1_SIZE_X;
    const int es = extra_begin[INPUT5_GET_INDEX(0, 0, 0, 1)];
    const int E = INPUT2_SIZE_X - es;
    const int rb = reset_begin[INPUT3_OFFSET];
    const int re = rb + reset_len[INPUT4_OFFSET];
    const int eb = extra_begin[INPUT5_OFFSET];
#    if HISTORY > 0
    const int C = INPUT6_SIZE_Y;
    const int T = INPUT6_SIZE_X;
    const int L = T + HISTORY;
#if FUSED_CONV
    if (i < (size_t)B * C * T) {
        // Prefill QKV is channel-major: adjacent lanes must read adjacent tokens.
        // Short decode batches favor contiguous [token,channel] output stores.
        const int c = T >= 16 ? (i / T) % C : i % C;
        const int t = T >= 16 ? i % T : (i / C) % T;
        const int b = i / (T * C);
        const int row = active_rows[INPUT1_GET_INDEX(0, 0, 0, b)];
        float sum = 0.0f;
        for (int j = 0; j <= HISTORY; ++j) {
            const int x = t + j;
            float v = x < HISTORY ? (row >= rb && row < re ? 0.0f :
                convert_float(old_cache[INPUT0_GET_INDEX(0, 0, row, c * HISTORY + x)])) :
                convert_float(qkv[INPUT6_GET_INDEX(0, b, c, x - HISTORY)]);
            v = convert_float(TO_OUTPUT_TYPE(v));
            sum = fma(v, convert_float(weights[INPUT8_GET_INDEX(0, 0, c, j)]), sum);
        }
        active[OUTPUT_GET_INDEX(0, b, t, c)] = TO_OUTPUT_TYPE(sum);
    }
#else
    if (i < (size_t)B * C * L) {
        const int t = i % L, c = (i / L) % C, b = i / (L * C);
        const int row = active_rows[INPUT1_GET_INDEX(0, 0, 0, b)];
        const float v = t < HISTORY ? (row >= rb && row < re ? 0.0f : convert_float(old_cache[INPUT0_GET_INDEX(0, 0, row, c * HISTORY + t)]))
                                    : convert_float(qkv[INPUT6_GET_INDEX(0, b, c, t - HISTORY)]);
        active[OUTPUT_GET_INDEX(0, b, c, t)] = TO_OUTPUT_TYPE(v);
    }
#endif
#    else
    if (i < (size_t)B * F) {
        const int f = i % F, b = i / F;
        const int row = active_rows[INPUT1_GET_INDEX(0, 0, 0, b)];
#        if STATE_KEY_SIZE > 0
        const int matrix = STATE_KEY_SIZE * STATE_VALUE_SIZE;
        const int sf = (f / matrix) * matrix + (f % STATE_VALUE_SIZE) * STATE_KEY_SIZE + (f / STATE_VALUE_SIZE) % STATE_KEY_SIZE;
#        else
        const int sf = f;
#        endif
        active[OUTPUT_GET_INDEX(0, 0, b, f)] = row >= rb && row < re ? (OUTPUT_TYPE)0 : old_cache[INPUT0_GET_INDEX(0, 0, row, sf)];
    }
#    endif
#if PREPARE_CACHE
    if (i < (size_t)S * F) {
        const int f = i % F, dst = i / F;
        int row = dst;
        if (dst >= eb && dst < eb + E)
            row = extra_rows[INPUT2_GET_INDEX(0, 0, 0, dst - eb + es)];
        float v = row >= rb && row < re ? 0.0f : convert_float(old_cache[INPUT0_GET_INDEX(0, 0, row, f)]);
#    if FUSED_CONV
        v = convert_float(TO_OUTPUT_TYPE(v));
#    endif
#    if HISTORY > 0
        // Later windows have the same precedence as the original ordered CPYs.
        for (int k = 0; k < INPUT7_SIZE_Y; ++k) {
            const int start = windows[INPUT7_GET_INDEX(0, 0, k, 0)];
            const int dest = windows[INPUT7_GET_INDEX(0, 0, k, 1)];
            if (dst >= dest && dst < dest + B) {
                const int b = dst - dest, c = f / HISTORY, t = start + f % HISTORY;
                const int src = active_rows[INPUT1_GET_INDEX(0, 0, 0, b)];
                v = t < HISTORY ? (src >= rb && src < re ? 0.0f : convert_float(old_cache[INPUT0_GET_INDEX(0, 0, src, c * HISTORY + t)]))
                                : convert_float(qkv[INPUT6_GET_INDEX(0, b, c, t - HISTORY)]);
                // Cache the materialized convolution input, including its precision conversion.
                v = convert_float(TO_OUTPUT_TYPE(v));
            }
        }
#    endif
        updated[OUTPUT1_GET_INDEX(0, 0, dst, f)] = TO_OUTPUT1_TYPE(v);
    }
#endif
}

#endif
