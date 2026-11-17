// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define CHANNELS_PER_GROUP (INPUT0_FEATURE_NUM / NUM_GROUPS)
#define SPATIAL_SIZE (INPUT0_SIZE_W * INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X)

// Index macros with all six coordinates. Unused coordinates are dropped for the tensor rank.
#if INPUT0_DIMS == 6
    #define INPUT_INDEX(b, f, w, z, y, x) INPUT0_GET_INDEX(b, f, w, z, y, x)
    #define OUT_INDEX(b, f, w, z, y, x) OUTPUT_GET_INDEX(b, f, w, z, y, x)
#elif INPUT0_DIMS == 5
    #define INPUT_INDEX(b, f, w, z, y, x) INPUT0_GET_INDEX(b, f, z, y, x)
    #define OUT_INDEX(b, f, w, z, y, x) OUTPUT_GET_INDEX(b, f, z, y, x)
#else
    #define INPUT_INDEX(b, f, w, z, y, x) INPUT0_GET_INDEX(b, f, y, x)
    #define OUT_INDEX(b, f, w, z, y, x) OUTPUT_GET_INDEX(b, f, y, x)
#endif

#ifdef GROUP_NORM_KERNEL_CALC_STATS

// f_begin: group index * CHANNELS_PER_GROUP
// i: linear index in the group with x innermost.
inline uint FUNC(group_element_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f_begin, uint i) {
    const uint x = i % INPUT0_SIZE_X;
    i /= INPUT0_SIZE_X;
    const uint y = i % INPUT0_SIZE_Y;
    i /= INPUT0_SIZE_Y;
    const uint z = i % INPUT0_SIZE_Z;
    i /= INPUT0_SIZE_Z;
    const uint w = i % INPUT0_SIZE_W;
    const uint f = f_begin + i / INPUT0_SIZE_W;
    return INPUT_INDEX(b, f, w, z, y, x);
}

// Computes the statistics of each (b, group) for the normalize stage:
//   mean_buf[b * NUM_GROUPS + g]    = mean of the group
//   inv_std_buf[b * NUM_GROUPS + g] = 1 / sqrt(variance + EPSILON)
// A group is CHANNELS_PER_GROUP channels x all spatial elements of batch b.
// Accumulates in f32 with a corrected two-pass algorithm (see below).
// GWS = {LWS * NUM_GROUPS, B, 1}, LWS = {LWS, 1, 1}. One work-group handles one (b, group).
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(calc_stats)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global float* mean_buf,
    __global float* inv_std_buf
) {
    const uint g = get_group_id(0);
    const uint b = get_global_id(1);
    const uint lid = get_local_id(0);
    const uint f_begin = g * CHANNELS_PER_GROUP;
    const uint n = CHANNELS_PER_GROUP * SPATIAL_SIZE;

    // Only for empty tensors. The condition is uniform in the work-group.
    if (n == 0 || b >= INPUT0_BATCH_NUM)
        return;

    // Pass 1: approx_mean = sum(x) / n
    // Kahan summation keeps rounding errors that accumulate in the same direction from biasing the mean.
    float sum = 0.0f;
    float comp = 0.0f;
    for (ulong i = lid; i < n; i += LWS) {
        const float x = convert_float(input[FUNC_CALL(group_element_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f_begin, (uint)i)]);
        // Kahan summation of sum += x. comp keeps the low-order bits lost by each addition.
        const float y = x - comp;
        const float t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    const float approx_mean = work_group_reduce_add(sum) / n;

    // Pass 2: d = x - approx_mean, mean = approx_mean + sum(d) / n, variance = (sum(d^2) - sum(d)^2 / n) / n
    // Corrected two-pass: the second pass also sums the deviations. That sum is zero in exact arithmetic,
    // so it measures the rounding error of approx_mean. It fixes the mean and removes the error from the variance.
    // Kahan summation preserves sum(d) = n * (mean - approx_mean), the correction term of the mean.
    // The deviations share its sign, so a plain f32 sum loses it.
    float d_sum = 0.0f;
    float d_comp = 0.0f;
    float sq_sum = 0.0f;
    for (ulong i = lid; i < n; i += LWS) {
        const float d = convert_float(input[FUNC_CALL(group_element_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f_begin, (uint)i)]) - approx_mean;
        // Kahan summation of d_sum += d. d_comp keeps the low-order bits lost by each addition.
        const float y = d - d_comp;
        const float t = d_sum + y;
        d_comp = (t - d_sum) - y;
        d_sum = t;
        sq_sum = fma(d, d, sq_sum);
    }
    d_sum = work_group_reduce_add(d_sum);
    sq_sum = work_group_reduce_add(sq_sum);
    const float mean = approx_mean + d_sum / n;
    // d_sum * d_sum can overflow while sq_sum is finite, so divide by n first.
    const float variance = fmax((sq_sum - d_sum * (d_sum / n)) / n, 0.0f);

    // Output: mean, inv_std = rsqrt(variance + EPSILON)
    if (lid == 0) {
        mean_buf[b * NUM_GROUPS + g] = mean;
        inv_std_buf[b * NUM_GROUPS + g] = rsqrt(variance + EPSILON);
    }
}

#elif defined(GROUP_NORM_KERNEL_NORMALIZE)

// GWS = {W * Z * Y * X, F, B}
KERNEL(normalize)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* scale,
    const __global INPUT2_TYPE* bias,
    const __global float* mean_buf,
    const __global float* inv_std_buf,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    uint s = get_global_id(0);
    const uint f = get_global_id(1);
    const uint b = get_global_id(2);

    // Only for empty tensors, where GWS is clamped to 1.
    if (s >= SPATIAL_SIZE || f >= INPUT0_FEATURE_NUM || b >= INPUT0_BATCH_NUM)
        return;

    const uint x = s % INPUT0_SIZE_X;
    s /= INPUT0_SIZE_X;
    const uint y = s % INPUT0_SIZE_Y;
    s /= INPUT0_SIZE_Y;
    const uint z = s % INPUT0_SIZE_Z;
    const uint w = s / INPUT0_SIZE_Z;

    const uint bg = b * NUM_GROUPS + f / CHANNELS_PER_GROUP;
    // scale and bias hold C contiguous values. Their layout may be [C] or [1, C, 1, 1], so read them linearly.
    float normalized = (convert_float(input[INPUT_INDEX(b, f, w, z, y, x)]) - mean_buf[bg]) * inv_std_buf[bg];
    normalized = normalized * convert_float(scale[f]) + convert_float(bias[f]);

    const uint output_idx = OUT_INDEX(b, f, w, z, y, x);
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = FUSED_OPS_RESULT;
#else
    output[output_idx] = TO_OUTPUT_TYPE(normalized);
#endif
}

#endif

#undef CHANNELS_PER_GROUP
#undef SPATIAL_SIZE
#undef INPUT_INDEX
#undef OUT_INDEX
