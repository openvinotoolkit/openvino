// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#ifdef GROUP_NORM_KERNEL_FEATURE_MEAN
KERNEL(calc_mean_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_mean
) {
    const uint bf = get_global_id(2);     // batch * feature
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;

    #if IS_DYNAMIC
        const uint y_num_workers = get_local_size(1);
        const uint x_num_workers = get_local_size(0);
    #else
        const uint y_num_workers = Y_NUM_WORKERS;
        const uint x_num_workers = X_NUM_WORKERS;
    #endif
    const uint y_block_size = INPUT0_SIZE_Y / y_num_workers;
    const uint y_base = get_local_id(1) * y_block_size;
    const uint y_leftover = INPUT0_SIZE_Y - y_num_workers * y_block_size;

    const uint x_block_size = INPUT0_SIZE_X / x_num_workers;
    const uint x_base = get_local_id(0);
    const uint x_leftover = INPUT0_SIZE_X - x_num_workers * x_block_size;

    const uint num_local_workers = y_num_workers * x_num_workers;
    const uint worker_idx = get_local_linear_id();

    __local ACCUMULATOR_TYPE mean_per_feature[SLM_SIZE];

    ACCUMULATOR_TYPE mean = ACCUMULATOR_VAL_ZERO;

    for (uint y = y_base; y < (y_base + y_block_size); ++y) {
        uint my_data_offset = INPUT0_GET_INDEX(b, f, y, x_base);
        for (uint i = 0; i < x_block_size; ++i) {
            mean += TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]);
        }
    }

    if (get_local_id(1) < y_leftover) {
        uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size), x_base);
        for (uint i = 0; i < x_block_size; ++i) {
            mean += TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]);
        }
    }

    if (get_local_id(0) < x_leftover) {
        for (uint y = y_base; y < (y_base + y_block_size); ++y) {
            uint my_data_offset = INPUT0_GET_INDEX(b, f, y, (get_local_id(0) + x_num_workers * x_block_size));
            mean += TO_ACCUMULATOR_TYPE(input[my_data_offset]);
        }
    }

    if (get_local_id(1) < y_leftover && get_local_id(0) < x_leftover) {
        uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size),
                                                     (get_local_id(0) + x_num_workers * x_block_size));
        mean += TO_ACCUMULATOR_TYPE(input[my_data_offset]);
    }

    mean_per_feature[worker_idx] = mean;
    uint reduce_add_level = 1;
    while (num_local_workers > reduce_add_level) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (worker_idx % (reduce_add_level * 2) == 0 && (worker_idx + reduce_add_level) < num_local_workers) {
            mean_per_feature[worker_idx] += mean_per_feature[worker_idx + reduce_add_level];
        }
        reduce_add_level *= 2;
    }

    if (worker_idx == 0) {
        mean = mean_per_feature[0] / TO_ACCUMULATOR_TYPE(INPUT0_SIZE_Y * INPUT0_SIZE_X);
        internal_mean[bf] = mean;
    }
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
KERNEL(calc_var_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_mean,
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint bf = get_global_id(2);     // batch * feature
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;

    #if IS_DYNAMIC
        const uint y_num_workers = get_local_size(1);
        const uint x_num_workers = get_local_size(0);
    #else
        const uint y_num_workers = Y_NUM_WORKERS;
        const uint x_num_workers = X_NUM_WORKERS;
    #endif
    const uint y_block_size = INPUT0_SIZE_Y / y_num_workers;
    const uint y_base = get_local_id(1) * y_block_size;
    const uint y_leftover = INPUT0_SIZE_Y - y_num_workers * y_block_size;

    const uint x_block_size = INPUT0_SIZE_X / x_num_workers;
    const uint x_base = get_local_id(0);
    const uint x_leftover = INPUT0_SIZE_X - x_num_workers * x_block_size;

    __local ACCUMULATOR_TYPE var_per_feature[SLM_SIZE];

    const ACCUMULATOR_TYPE mean = internal_mean[bf];
    ACCUMULATOR_TYPE variance = ACCUMULATOR_VAL_ZERO;

    for (uint y = y_base; y < (y_base + y_block_size); ++y) {
        uint my_data_offset = INPUT0_GET_INDEX(b, f, y, x_base);
        for (uint i = 0; i < x_block_size; ++i) {
            ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]);
            tmp -= mean;
            variance = fma(tmp, tmp, variance);
        }
    }

    if (get_local_id(1) < y_leftover) {
        uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size), x_base);
        for (uint i = 0; i < x_block_size; ++i) {
            ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]);
            tmp -= mean;
            variance = fma(tmp, tmp, variance);
        }
    }

    if (get_local_id(0) < x_leftover) {
        for (uint y = y_base; y < (y_base + y_block_size); ++y) {
            uint my_data_offset = INPUT0_GET_INDEX(b, f, y, (get_local_id(0) + x_num_workers * x_block_size));
            ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[my_data_offset]);
            tmp -= mean;
            variance = fma(tmp, tmp, variance);
        }
    }

    if (get_local_id(1) < y_leftover && get_local_id(0) < x_leftover) {
        uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size),
                                                     (get_local_id(0) + x_num_workers * x_block_size));
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[my_data_offset]);
        tmp -= mean;
        variance = fma(tmp, tmp, variance);
    }

    const uint num_local_workers = y_num_workers * x_num_workers;
    const uint worker_idx = get_local_linear_id();

    var_per_feature[worker_idx] = variance;
    uint reduce_add_level = 1;
    while (num_local_workers > reduce_add_level) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (worker_idx % (reduce_add_level * 2) == 0 && (worker_idx + reduce_add_level) < num_local_workers) {
            var_per_feature[worker_idx] += var_per_feature[worker_idx + reduce_add_level];
        }
        reduce_add_level *= 2;
    }

    if (worker_idx == 0) {
        variance = var_per_feature[0] / TO_ACCUMULATOR_TYPE(INPUT0_SIZE_Y * INPUT0_SIZE_X);
        internal_variance[bf] = variance;
    }
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
    const uint bf = get_global_id(1);
    const uint b = bf / OUTPUT_FEATURE_NUM;
    const uint f = bf % OUTPUT_FEATURE_NUM;
    const uint yx = get_global_id(0);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;

    const uint input_data_index = INPUT0_GET_INDEX(b, f, y, x);

    ACTIVATION_TYPE mean = TO_ACTIVATION_TYPE(internal_mean[bf]);
    ACTIVATION_TYPE variance = TO_ACTIVATION_TYPE(internal_variance[bf]);
    ACTIVATION_TYPE normalized = (TO_ACTIVATION_TYPE(input[input_data_index]) - mean) * variance;
    normalized = normalized * TO_ACTIVATION_TYPE(scale[f]) + TO_ACTIVATION_TYPE(bias[f]);

    const uint output_data_index = OUTPUT_GET_INDEX(b, f, y, x);
    #if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_data_index] = FUSED_OPS_RESULT;
    #else
        output[output_data_index] = TO_OUTPUT_TYPE(ACTIVATION(normalized, ACTIVATION_PARAMS));
    #endif
}
#endif
