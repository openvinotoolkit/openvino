// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#ifdef GROUP_NORM_KERNEL_FEATURE_MEAN_SQR_MEAN
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(calc_mean_sqr_mean_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_mean,
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint b = get_global_id(2) / INPUT0_FEATURE_NUM;
    const uint f = get_global_id(2) % INPUT0_FEATURE_NUM;
    const uint y = get_global_id(1);
    const uint x = get_global_id(0);
    const uint divisor_x = INPUT0_SIZE_X / get_local_size(0);
    const uint divisor_y = INPUT0_SIZE_Y / get_local_size(1);

    ACCUMULATOR_TYPE local_sum = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE local_sqr_sum = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE wi_sum = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE wi_sqr_sum = ACCUMULATOR_VAL_ZERO;
    unroll_for (uint i = 0; i < divisor_y; ++i) {
        unroll_for (uint j = 0; j < divisor_x; ++j) {
            const uint data_offset = INPUT0_GET_INDEX(b, f, y + (get_local_size(1) * i), x + (get_local_size(0) * j));
            ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[data_offset]);
            wi_sum += data;
            wi_sqr_sum += data * data;
        }
    }

    local_sum += work_group_reduce_add(wi_sum);
    local_sqr_sum += work_group_reduce_add(wi_sqr_sum);

    uint bf = b * INPUT0_FEATURE_NUM + f;
    if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
        uint group_size = get_num_groups(0) * get_num_groups(1) * get_num_groups(2);
        uint group_wi_size = INPUT0_SIZE_X * INPUT0_SIZE_Y;
        float mean = local_sum / TO_ACCUMULATOR_TYPE(group_wi_size);
        float variance = local_sqr_sum / TO_ACCUMULATOR_TYPE(group_wi_size);
        internal_mean[b * INPUT0_FEATURE_NUM + f] = mean;
        internal_variance[b * INPUT0_FEATURE_NUM + f] = variance;
    }
}
#elif GROUP_NORM_KERNEL_GROUP_MEAN_VARIANCE
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(calc_mean_variance_per_group)(
    __global ACCUMULATOR_TYPE* internal_mean,
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint data_idx = get_global_id(0) + get_global_id(1) * GWS0;
    const uint num_workers = LWS0;
    const uint group_size = GWS0 / NUM_GROUPS;
    const uint items_num = group_size / num_workers;

    if ((data_idx % group_size) < num_workers) {
        ACCUMULATOR_TYPE mean_sum = ACCUMULATOR_VAL_ZERO;
        ACCUMULATOR_TYPE variance_sum = ACCUMULATOR_VAL_ZERO;
        for (uint i = 0; i < items_num; ++i) {
            mean_sum += internal_mean[data_idx + num_workers * i];
            variance_sum += internal_variance[data_idx + num_workers * i];
        }

        ACCUMULATOR_TYPE mean = work_group_reduce_add(mean_sum);
        ACCUMULATOR_TYPE variance = work_group_reduce_add(variance_sum);
        mean /= TO_ACCUMULATOR_TYPE(group_size);
        variance /= TO_ACCUMULATOR_TYPE(group_size);
        variance -=  mean * mean;
        variance = native_powr(variance + TO_ACCUMULATOR_TYPE(EPSILON), -0.5f);
        for (uint i = 0; i < items_num; ++i) {
            internal_mean[data_idx + num_workers * i] = mean;
            internal_variance[data_idx + num_workers * i] = variance;
        }
    }
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
    const __global ACCUMULATOR_TYPE* internal_mean,
    const __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint b = get_global_id(1) % OUTPUT_BATCH_NUM;
    const uint f = get_global_id(1) / OUTPUT_BATCH_NUM * FSV + (get_sub_group_local_id() % FSV);
    const uint yx = get_global_id(0) / FSV;
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint input_index = INPUT0_GET_INDEX(b, f, y, x);
    const uint output_index = OUTPUT_GET_INDEX(b, f, y, x);

    if (f < OUTPUT_FEATURE_NUM) {
        const uint bf = b * OUTPUT_FEATURE_NUM + f;
        ACTIVATION_TYPE mean = TO_ACTIVATION_TYPE(internal_mean[bf]);
        ACTIVATION_TYPE variance = TO_ACTIVATION_TYPE(internal_variance[bf]);
        ACTIVATION_TYPE normalized = (TO_ACTIVATION_TYPE(input[input_index]) - mean) * variance;
        normalized = normalized * TO_ACTIVATION_TYPE(scale[f]) + TO_ACTIVATION_TYPE(bias[f]);
        #if HAS_FUSED_OPS
            FUSED_OPS;
            output[output_index] = FUSED_OPS_RESULT;
        #else
            output[output_index] = TO_OUTPUT_TYPE(normalized);
        #endif
    } else {
        #ifdef OUTPUT_LAYOUT_B_FS_YX_FSV16
            output[output_index] = OUTPUT_VAL_ZERO;
        #endif
    }
}
#endif
