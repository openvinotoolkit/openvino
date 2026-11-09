// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#define DATA_COUNT ((uint)(INPUT0_BATCH_NUM * INPUT0_FEATURE_NUM * INPUT0_SIZE_V * INPUT0_SIZE_U * INPUT0_SIZE_W * INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X))
#define SG_CHUNK (SIMD * VEC_SIZE)

// Runs BODY(x) for every input element assigned to this work-item. Sub-groups stride over SG_CHUNK-element chunks
// read with block reads (the element order does not matter for a histogram); the remainder is read per work-item.
#define FOR_EACH_INPUT(BODY)                                                                                       \
    {                                                                                                              \
        const uint sg_id = get_group_id(0) * get_num_sub_groups() + get_sub_group_id();                            \
        const uint sg_num = get_num_groups(0) * get_num_sub_groups();                                              \
        for (uint c = sg_id; c < DATA_COUNT / SG_CHUNK; c += sg_num) {                                             \
            const MAKE_VECTOR_TYPE(INPUT0_TYPE, VEC_SIZE) vals = BLOCK_READN(INPUT0_TYPE, VEC_SIZE, data, c * SG_CHUNK); \
            unroll_for(uint k = 0; k < VEC_SIZE; ++k) {                                                            \
                const float x = convert_float(vals[k]);                                                            \
                BODY(x);                                                                                           \
            }                                                                                                      \
        }                                                                                                          \
    }                                                                                                              \
    for (uint i = DATA_COUNT / SG_CHUNK * SG_CHUNK + get_global_id(0); i < DATA_COUNT; i += get_global_size(0)) { \
        const float x = convert_float(data[i]);                                                                    \
        BODY(x);                                                                                                   \
    }

#if HISTC_STAGE_MINMAX
// Every work-group reduces a strided part of the input to a partial [min, max] pair.
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(histc_minmax)(OPTIONAL_SHAPE_INFO_ARG
                     const __global INPUT0_TYPE* restrict data,
                     __global float* restrict ranges) {
    float lo = INFINITY;
    float hi = -INFINITY;
// fmin/fmax drop NaN, so a NaN widens the range to infinity, which is rejected like an infinite range.
#define UPDATE_RANGE(x) lo = isnan(x) ? -INFINITY : fmin(lo, x); hi = isnan(x) ? INFINITY : fmax(hi, x)
    FOR_EACH_INPUT(UPDATE_RANGE)
#undef UPDATE_RANGE
    lo = work_group_reduce_min(lo);
    hi = work_group_reduce_max(hi);
    if (get_local_id(0) == 0) {
        ranges[2 * get_group_id(0)] = lo;
        ranges[2 * get_group_id(0) + 1] = hi;
    }
}
#endif

#if HISTC_STAGE_COUNT
// Every work-group counts a strided part of the input into its own partial histogram.
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(histc_count)(OPTIONAL_SHAPE_INFO_ARG
                    const __global INPUT0_TYPE* restrict data,
                    __global uint* restrict partial_hist
#if RANGE_FROM_DATA
                    , const __global float* restrict ranges
#endif
) {
    const uint lid = get_local_id(0);
    const uint lws = get_local_size(0);

#if RANGE_FROM_DATA
    float lo = INFINITY;
    float hi = -INFINITY;
    for (uint g = lid; g < get_num_groups(0); g += lws) {
        lo = fmin(lo, ranges[2 * g]);
        hi = fmax(hi, ranges[2 * g + 1]);
    }
    lo = work_group_reduce_min(lo);
    hi = work_group_reduce_max(hi);
    if (lo == hi) {
        // The spec widens with nextafter only for f32 (and f64, which runs as f32).
#if INPUT0_IS_FP && INPUT0_TYPE_SIZE == 4
        lo = fmin(nextafter(lo, -INFINITY), lo - 1.0f);
        hi = fmax(nextafter(hi, INFINITY), hi + 1.0f);
#else
        lo -= 1.0f;
        hi += 1.0f;
#endif
    }
#else
    const float lo = RANGE_MIN;
    const float hi = RANGE_MAX;
#endif

#if HIST_IN_SLM
    // Interleaved copies: neighbouring work-items that hit the same bin use different SLM banks.
    __local uint hist[BINS * HIST_COPIES];
    const uint copy = lid % HIST_COPIES;
    #define HIST_SIZE (BINS * HIST_COPIES)
    #define HIST_MEM_FENCE CLK_LOCAL_MEM_FENCE
#else
    __global uint* hist = partial_hist + get_group_id(0) * BINS;
    const uint copy = 0;
    #define HIST_SIZE BINS
    #define HIST_MEM_FENCE CLK_GLOBAL_MEM_FENCE
#endif
    for (uint b = lid; b < HIST_SIZE; b += lws) {
        hist[b] = 0;
    }
    barrier(HIST_MEM_FENCE);

    // An invalid range (e.g. taken from data with infinities) yields an empty histogram.
    if (isfinite(lo) && isfinite(hi) && lo < hi) {
        const float range = hi - lo;
#define COUNT_ELEMENT(x)                                                                          \
        if (x >= lo && x <= hi) {                                                                 \
            const uint bin = min(convert_uint((x - lo) * (float)BINS / range), (uint)(BINS - 1)); \
            atomic_inc(&hist[bin * HIST_COPIES + copy]);                                          \
        }
        FOR_EACH_INPUT(COUNT_ELEMENT)
#undef COUNT_ELEMENT
    }

#if HIST_IN_SLM
    barrier(HIST_MEM_FENCE);
    for (uint b = lid; b < BINS; b += lws) {
        uint count = 0;
        unroll_for(uint c = 0; c < HIST_COPIES; ++c) {
            count += hist[b * HIST_COPIES + c];
        }
        partial_hist[get_group_id(0) * BINS + b] = count;
    }
#endif
#undef HIST_SIZE
#undef HIST_MEM_FENCE
}
#endif

#if HISTC_STAGE_MERGE
KERNEL(histc_merge)(const __global uint* restrict partial_hist,
                    __global OUTPUT_TYPE* restrict output,
                    const uint num_groups) {
    const uint bin = get_global_id(0);
    if (bin >= BINS)
        return;

    uint count = 0;
    for (uint g = 0; g < num_groups; ++g) {
        count += partial_hist[g * BINS + bin];
    }
    // Counts not representable in the output type are implementation defined, so skip saturation.
    output[bin] = TO_OUTPUT_TYPE(count);
}
#endif

#undef SG_CHUNK
#undef FOR_EACH_INPUT
#undef DATA_COUNT
