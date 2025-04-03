// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#define NUM_CHANNELS_IN_GROUP (INPUT0_FEATURE_NUM / NUM_GROUPS)
#define CHANNEL_SIZE (INPUT0_BATCH_PITCH / INPUT0_FEATURE_NUM)
#define GROUP_SIZE (NUM_CHANNELS_IN_GROUP * CHANNEL_SIZE)

#if MEAN_KERNEL_ENABLED || STANDARD_DEVIATION_KERNEL_ENABLED
inline void FUNC(kahan_summation)(INPUT0_TYPE elem, __private float* compensation, __private float* sum) {
    if (isfinite(elem) && isfinite(*sum)) {
        float temp = *sum + (elem - *compensation);
        *compensation = (temp - *sum) - (elem - *compensation);
        *sum = temp;
    } else {
        *sum += elem;
    }
}
#endif

#if MEAN_KERNEL_ENABLED

KERNEL (calc_mean_ref)(  __global INPUT0_TYPE* input
                       , __global float* output
#if HAS_FUSED_OPS_DECLS
                       , FUSED_OPS_DECLS
#endif
)
{
    const int batch = get_global_id(0);
    if (batch >= INPUT0_BATCH_NUM)
        return;
    const int group = get_global_id(1);
    const int feature_begin = group * NUM_CHANNELS_IN_GROUP;
    const int feature_end = group * NUM_CHANNELS_IN_GROUP + NUM_CHANNELS_IN_GROUP;
    float variance = 0.f, error = 0.f, mean_value = 0.f;
    for (int feature = feature_begin; feature < feature_end; feature++)
    {
        if (feature >= INPUT0_FEATURE_NUM)
            continue;
#if OUTPUT_DIMS > 4
        for (int z = 0; z < INPUT0_SIZE_Z; z++)
#endif
        for (int y = 0; y < INPUT0_SIZE_Y; y++)
            for (int x = 0; x < INPUT0_SIZE_X; x++)
            {
#if OUTPUT_DIMS == 5
                size_t input_idx = INPUT0_GET_INDEX(batch, feature, z, y, x);
#elif OUTPUT_DIMS == 4
                size_t input_idx = INPUT0_GET_INDEX(batch, feature, y, x);
#endif
                FUNC_CALL(kahan_summation)(input[input_idx], &error, &mean_value);
            }
    }
    mean_value /= GROUP_SIZE;
    output[batch * NUM_GROUPS + group] = mean_value;
}

#elif STANDARD_DEVIATION_KERNEL_ENABLED

KERNEL (calc_standard_deviation_ref)(  __global INPUT0_TYPE* input
                                     , __global float* mean
                                     , __global float* output
#if HAS_FUSED_OPS_DECLS
                                     , FUSED_OPS_DECLS
#endif
)
{
    const int batch = get_global_id(0);
    if (batch >= INPUT0_BATCH_NUM)
        return;
    const int group = get_global_id(1);
    const output_idx = batch * NUM_GROUPS + group;
    const int feature_begin = group * NUM_CHANNELS_IN_GROUP;
    const int feature_end = group * NUM_CHANNELS_IN_GROUP + NUM_CHANNELS_IN_GROUP;
    float variance = 0.f, error = 0.f;

    for (int feature = feature_begin; feature < feature_end; feature++)
    {
        if (feature >= INPUT0_FEATURE_NUM)
            continue;
#if OUTPUT_DIMS > 4
        for (int z = 0; z < INPUT0_SIZE_Z; z++)
#endif
        for (int y = 0; y < INPUT0_SIZE_Y; y++)
            for (int x = 0; x < INPUT0_SIZE_X; x++)
            {
#if OUTPUT_DIMS == 5
                size_t input_idx = INPUT0_GET_INDEX(batch, feature, z, y, x);
#elif OUTPUT_DIMS == 4
                size_t input_idx = INPUT0_GET_INDEX(batch, feature, y, x);
#endif
                FUNC_CALL(kahan_summation)(pow(input[input_idx] - mean[output_idx], 2), &error, &variance);
            }
    }
    variance /= GROUP_SIZE;
    float standard_deviation = sqrt(variance + EPSILON);
    output[output_idx] = standard_deviation;
}
#elif NORMALIZE_KERNEL_ENABLED
KERNEL (normalize_ref)(  __global INPUT0_TYPE* input
                       , __global INPUT0_TYPE* scale_values
                       , __global INPUT0_TYPE* bias_values
                       , __global float* mean_values
                       , __global float* standard_deviation_values
                       , __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                       , FUSED_OPS_DECLS
#endif
)
{
    const int batch = get_global_id(0);
#if OUTPUT_DIMS == 4
    const int feature = get_global_id(1);
#elif OUTPUT_DIMS == 5
    const int feature = get_global_id(1) / OUTPUT_SIZE_Z;
    const int z = get_global_id(1) % OUTPUT_SIZE_Z;
#endif
    const int y = get_global_id(2) / OUTPUT_SIZE_X;
    const int x = get_global_id(2) % OUTPUT_SIZE_X;
    const int group = feature / NUM_CHANNELS_IN_GROUP;
    float mean = mean_values[batch * NUM_GROUPS + group];
    float standard_deviation = standard_deviation_values[batch * NUM_GROUPS + group];
#if OUTPUT_DIMS == 4
    size_t output_idx = OUTPUT_GET_INDEX(batch, feature, y, x);
#elif OUTPUT_DIMS == 5
    size_t output_idx = OUTPUT_GET_INDEX(batch, feature, z, y, x);
#endif
    OUTPUT_TYPE res = ((input[output_idx] - mean) / standard_deviation) * scale_values[feature] + bias_values[feature];
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = FUSED_OPS_RESULT;
#else
    output[output_idx] = res;
#endif
}

#endif

#undef NUM_CHANNELS_IN_GROUP
#undef CHANNEL_SIZE
#undef GROUP_SIZE
