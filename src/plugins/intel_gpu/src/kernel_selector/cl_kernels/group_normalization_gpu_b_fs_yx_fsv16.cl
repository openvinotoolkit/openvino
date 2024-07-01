// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#ifdef GROUP_NORM_KERNEL_FEATURE_MEAN
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(calc_mean_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_mean
) {
    const uint bfs = get_global_id(1);
    const uint in_data_set_idx = get_global_id(0);

    const uint data_set_offset = bfs * INPUT0_FEATURE_PITCH * FSV;
    const uint my_data_offset = data_set_offset + in_data_set_idx;

    ACCUMULATOR_TYPE mean = ACCUMULATOR_VAL_ZERO;

    for (uint i = 0; i < (INPUT0_SIZE_X * INPUT0_SIZE_Y); ++i) {
        mean += TO_ACCUMULATOR_TYPE(input[my_data_offset + i * SIMD]);
    }
    mean /= TO_ACCUMULATOR_TYPE(INPUT0_SIZE_X * INPUT0_SIZE_Y);
    uint bf = bfs * FSV + in_data_set_idx;
    internal_mean[bf] = mean;
}
#elif GROUP_NORM_KERNEL_GROUP_MEAN
KERNEL(calc_mean_per_group)(
    __global ACCUMULATOR_TYPE* internal_mean
) {
    const uint data_idx = get_global_id(0) + get_global_id(1) * get_global_size(0);
    const uint group_size = get_local_size(0);

    ACCUMULATOR_TYPE mean = work_group_reduce_add(internal_mean[data_idx]);
    mean /= TO_ACCUMULATOR_TYPE(group_size);
    internal_mean[data_idx] = mean;
}
#elif GROUP_NORM_KERNEL_FEATURE_VAR
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(calc_var_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_mean,
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint bfs = get_global_id(1);
    const uint in_data_set_idx = get_global_id(0);

    const uint data_set_offset = bfs * INPUT0_FEATURE_PITCH * FSV;
    const uint my_data_offset = data_set_offset + in_data_set_idx;
    uint bf = bfs * FSV + in_data_set_idx;

    ACCUMULATOR_TYPE mean = internal_mean[bf];
    ACCUMULATOR_TYPE variance = ACCUMULATOR_VAL_ZERO;

    for (uint i = 0; i < (INPUT0_SIZE_X * INPUT0_SIZE_Y); ++i) {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[my_data_offset + i * SIMD]);
        tmp -= mean;
        variance = fma(tmp, tmp, variance);
    }
    variance /= TO_ACCUMULATOR_TYPE(INPUT0_SIZE_X * INPUT0_SIZE_Y);
    internal_variance[bf] = variance;
}
#elif GROUP_NORM_KERNEL_GROUP_VAR
KERNEL(calc_var_per_group)(
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint data_idx = get_global_id(0) + get_global_id(1) * get_global_size(0);
    const uint group_size = get_local_size(0);

    ACCUMULATOR_TYPE variance = work_group_reduce_add(internal_variance[data_idx]);
    variance /= TO_ACCUMULATOR_TYPE(group_size);
    variance = native_powr(variance + TO_ACCUMULATOR_TYPE(EPSILON), -0.5f);
    internal_variance[data_idx] = variance;
}
#elif GROUP_NORM_KERNEL_FINAL
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(group_normalization_b_fs_yx_fsv16)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* scale,
    const __global INPUT2_TYPE* bias,
    __global OUTPUT_TYPE* restrict output,
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    __global ACCUMULATOR_TYPE* internal_mean,
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint bfs = get_global_id(1);
    const uint data_index = bfs * INPUT0_FEATURE_PITCH * FSV + get_global_id(0);
    uint bf = bfs * FSV + get_sub_group_local_id();
    uint feature_index = bf % OUTPUT_FEATURE_NUM;

    if (feature_index < INPUT0_FEATURE_NUM) {
        ACTIVATION_TYPE mean = TO_ACTIVATION_TYPE(internal_mean[bf]);
        ACTIVATION_TYPE variance = TO_ACTIVATION_TYPE(internal_variance[bf]);
        ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(input[data_index]) - mean) * variance;
        result = result * TO_ACTIVATION_TYPE(scale[feature_index]) + TO_ACTIVATION_TYPE(bias[feature_index]);
        #if HAS_FUSED_OPS
            FUSED_OPS;
            output[data_index] = FUSED_OPS_RESULT;
        #else
            output[data_index] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
        #endif
    } else {
        output[data_index] = ACTIVATION_VAL_ZERO;
    }
}
#endif