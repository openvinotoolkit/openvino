// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// Qwen llama.cpp staged gather: capture every original source needed by this op
// before any in-place cache writes. Metadata stays on device and varies each call.
#if INPLACE_STAGE == 0
KERNEL(recurrent_cache_update_stage)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* old_cache,
                                     const __global INPUT1_TYPE* active_rows,
                                     const __global INPUT2_TYPE* extra_rows,
                                     const __global INPUT3_TYPE* reset_begin,
                                     const __global INPUT4_TYPE* reset_length,
                                     const __global INPUT5_TYPE* offsets,
                                     const __global INPUT6_TYPE* qkv,
                                     const __global INPUT7_TYPE* windows,
                                     const __global INPUT8_TYPE* weights,
                                     __global OUTPUT_TYPE* output,
                                     __global OUTPUT1_TYPE* cache,
                                     __global float* staged,
                                     int inplace) {
    const size_t i = get_global_id(0);
    const int F = INPUT0_SIZE_X, S = INPUT0_SIZE_Y, B = INPUT1_SIZE_X;
    const int rb = reset_begin[INPUT3_OFFSET], re = rb + reset_length[INPUT4_OFFSET];
    const int es = offsets[INPUT5_GET_INDEX(0, 0, 0, 1)], eb = offsets[INPUT5_OFFSET], E = INPUT2_SIZE_X - es;
#    define OLD(r, f) old_cache[INPUT0_GET_INDEX(0, 0, r, f)]
#    define ROWA(b)   active_rows[INPUT1_GET_INDEX(0, 0, 0, b)]
#    define ROWE(e)   extra_rows[INPUT2_GET_INDEX(0, 0, 0, e + es)]
#    define ROUND(v)  convert_float((INPUT6_TYPE)(v))
    if (i >= (size_t)(B + E) * F)
        return;
    const int row = i / F, f = i % F, src = row < B ? ROWA(row) : ROWE(row - B);
    // Identity suffix rows are read by their destination owner in the compute
    // stage. They need no staging; all nonidentity sources are captured first.
    if (row >= B && src == eb + row - B)
        return;
    staged[i] = src >= rb && src < re ? 0.0f : ROUND(OLD(src, f));
#    undef OLD
#    undef ROWA
#    undef ROWE
#    undef ROUND
}
#else
// One destination owner combines convolution/cache writeback. In alias mode all
// cross-row reads use the gathered scratch, preserving cycles and duplicate rows.
KERNEL(recurrent_conv_cache_update)
(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* old_cache,
 const __global INPUT1_TYPE* active_rows,
 const __global INPUT2_TYPE* extra_rows,
 const __global INPUT3_TYPE* reset_begin,
 const __global INPUT4_TYPE* reset_len,
 const __global INPUT5_TYPE* extra_begin,
 const __global INPUT6_TYPE* qkv,
 const __global INPUT7_TYPE* windows,
 const __global INPUT8_TYPE* weights,
 __global OUTPUT_TYPE* active,
 __global OUTPUT1_TYPE* updated,
 __global float* staged,
 int inplace) {
    const size_t i = get_global_id(0);
    const int F = INPUT0_SIZE_X;
    const int S = INPUT0_SIZE_Y;
    const int B = INPUT1_SIZE_X;
    const int es = extra_begin[INPUT5_GET_INDEX(0, 0, 0, 1)];
    const int E = INPUT2_SIZE_X - es;
    const int rb = reset_begin[INPUT3_OFFSET];
    const int re = rb + reset_len[INPUT4_OFFSET];
    const int eb = extra_begin[INPUT5_OFFSET];
    const int C = INPUT6_SIZE_Y;
    const int T = INPUT6_SIZE_X;
    const int L = T + HISTORY;
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
            float v = x < HISTORY ? (row >= rb && row < re
                                         ? 0.0f
                                         : (inplace ? staged[b * F + c * HISTORY + x] : convert_float(old_cache[INPUT0_GET_INDEX(0, 0, row, c * HISTORY + x)])))
                                  : convert_float(qkv[INPUT6_GET_INDEX(0, b, c, x - HISTORY)]);
            v = convert_float(TO_OUTPUT_TYPE(v));
            sum = fma(v, convert_float(weights[INPUT8_GET_INDEX(0, 0, c, j)]), sum);
        }
        active[OUTPUT_GET_INDEX(0, b, t, c)] = TO_OUTPUT_TYPE(sum);
    }
    if (i < (size_t)S * F) {
        const int f = i % F, dst = i / F;
        int row = dst;
        if (dst >= eb && dst < eb + E)
            row = extra_rows[INPUT2_GET_INDEX(0, 0, 0, dst - eb + es)];
        float v =
            inplace && row != dst ? staged[(B + dst - eb) * F + f] : (row >= rb && row < re ? 0.0f : convert_float(old_cache[INPUT0_GET_INDEX(0, 0, row, f)]));
        v = convert_float(TO_OUTPUT_TYPE(v));
        // Later windows have the same precedence as the original ordered CPYs.
        for (int k = 0; k < INPUT7_SIZE_Y; ++k) {
            const int start = windows[INPUT7_GET_INDEX(0, 0, k, 0)];
            const int dest = windows[INPUT7_GET_INDEX(0, 0, k, 1)];
            if (dst >= dest && dst < dest + B) {
                const int b = dst - dest, c = f / HISTORY, t = start + f % HISTORY;
                const int src = active_rows[INPUT1_GET_INDEX(0, 0, 0, b)];
                v = t < HISTORY ? (inplace ? staged[b * F + c * HISTORY + t]
                                           : (src >= rb && src < re ? 0.0f : convert_float(old_cache[INPUT0_GET_INDEX(0, 0, src, c * HISTORY + t)])))
                                : convert_float(qkv[INPUT6_GET_INDEX(0, b, c, t - HISTORY)]);
                // Cache the materialized convolution input, including its precision conversion.
                v = convert_float(TO_OUTPUT_TYPE(v));
            }
        }
        // Each cache element has one owner. Canonical preserved rows need no
        // write, while reset/reorder/windows keep their original precedence.
        if (!inplace || as_uint(v) != as_uint(convert_float(old_cache[INPUT0_GET_INDEX(0, 0, dst, f)])))
            updated[OUTPUT1_GET_INDEX(0, 0, dst, f)] = TO_OUTPUT1_TYPE(v);
    }
}

#endif
