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
    const uint data_set_idx = get_global_id(1);     // batch * feature split
    const uint in_data_set_idx = get_global_id(0);
    const uint workers_per_dataset = LWS0 / FSV;    // 16 datasets are handled by one local workgroup
    const uint data_set_size = INPUT0_SIZE_X * INPUT0_SIZE_Y;
    const uint items_num = data_set_size / workers_per_dataset;
    const uint leftovers = data_set_size - (items_num * workers_per_dataset);

    const uint INPUT0_ALIGNED_FEATURE_NUM = ALIGN(INPUT0_FEATURE_NUM, FSV);
    const uint b = (data_set_idx * FSV) / INPUT0_ALIGNED_FEATURE_NUM;
    const uint f_base = (data_set_idx * FSV) % INPUT0_ALIGNED_FEATURE_NUM;
    const uint data_set_offset = INPUT0_GET_INDEX(b, f_base, 0, 0);
    const uint my_data_offset = data_set_offset + in_data_set_idx;

    __local ACCUMULATOR_TYPE mean_per_feature[SLM_SIZE];

    ACCUMULATOR_TYPE mean = ACCUMULATOR_VAL_ZERO;

    for (uint i = 0; i < items_num; ++i) {
        mean += TO_ACCUMULATOR_TYPE(input[my_data_offset + i * workers_per_dataset * FSV]);
    }

    if (in_data_set_idx < leftovers) {
        mean += TO_ACCUMULATOR_TYPE(input[my_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx]);
    }

    mean_per_feature[in_data_set_idx] = mean;
    const uint num_local_workers = LWS0;
    const uint worker_block_idx = in_data_set_idx / FSV;
    uint reduce_add_level = 1;
    while ((SLM_SIZE / FSV) > reduce_add_level) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (worker_block_idx % (reduce_add_level * 2) == 0 && (in_data_set_idx + FSV * reduce_add_level) < num_local_workers) {
            mean_per_feature[in_data_set_idx] += mean_per_feature[in_data_set_idx + FSV * reduce_add_level];
        }
        reduce_add_level *= 2;
    }

    if (worker_block_idx == 0 && (f_base + in_data_set_idx) < INPUT0_FEATURE_NUM) {
        mean = mean_per_feature[in_data_set_idx] / TO_ACCUMULATOR_TYPE(data_set_size);
        uint bf = b * INPUT0_FEATURE_NUM + f_base + in_data_set_idx;
        internal_mean[bf] = mean;
    }
}
#elif GROUP_NORM_KERNEL_GROUP_MEAN
KERNEL(calc_mean_per_group)(
    __global ACCUMULATOR_TYPE* internal_mean
) {
    const uint data_idx = get_global_id(0) + get_global_id(1) * GWS0;
    const uint num_workers = LWS0;
    const uint group_size = GWS0 / NUM_GROUPS;
    const uint items_num = group_size / num_workers;

    if ((data_idx % group_size) < num_workers) {
        ACCUMULATOR_TYPE my_sum = ACCUMULATOR_VAL_ZERO;
        for (uint i = 0; i < items_num; ++i) {
            my_sum += internal_mean[data_idx + num_workers * i];
        }

        ACCUMULATOR_TYPE mean = work_group_reduce_add(my_sum);
        mean /= TO_ACCUMULATOR_TYPE(group_size);
        for (uint i = 0; i < items_num; ++i) {
            internal_mean[data_idx + num_workers * i] = mean;
        }
    }
}
#elif GROUP_NORM_KERNEL_FEATURE_VAR
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(calc_var_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global ACCUMULATOR_TYPE* internal_mean,
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint data_set_idx = get_global_id(1);     // batch * feature split
    const uint in_data_set_idx = get_global_id(0);
    const uint workers_per_dataset = LWS0 / FSV;    // 16 datasets are handled by one local workgroup
    const uint data_set_size = INPUT0_SIZE_X * INPUT0_SIZE_Y;
    const uint items_num = data_set_size / workers_per_dataset;
    const uint leftovers = data_set_size - (items_num * workers_per_dataset);

    const uint INPUT0_ALIGNED_FEATURE_NUM = ALIGN(INPUT0_FEATURE_NUM, FSV);
    const uint b = (data_set_idx * FSV) / INPUT0_ALIGNED_FEATURE_NUM;
    const uint f_base = (data_set_idx * FSV) % INPUT0_ALIGNED_FEATURE_NUM;
    const uint data_set_offset = INPUT0_GET_INDEX(b, f_base, 0, 0);
    const uint my_data_offset = data_set_offset + in_data_set_idx;

    __local ACCUMULATOR_TYPE var_per_feature[SLM_SIZE];

    uint bf = b * INPUT0_FEATURE_NUM + f_base + get_sub_group_local_id();

    ACCUMULATOR_TYPE mean = internal_mean[bf];
    ACCUMULATOR_TYPE variance = ACCUMULATOR_VAL_ZERO;

    for (uint i = 0; i < items_num; ++i) {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[my_data_offset + i * workers_per_dataset * FSV]);
        tmp -= mean;
        variance = fma(tmp, tmp, variance);
    }

    if (in_data_set_idx < leftovers) {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[my_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx]);
        tmp -= mean;
        variance = fma(tmp, tmp, variance);
    }

    var_per_feature[in_data_set_idx] = variance;
    const uint num_local_workers = LWS0;
    const uint worker_block_idx = in_data_set_idx / FSV;
    uint reduce_add_level = 1;
    while ((SLM_SIZE / FSV) > reduce_add_level) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (worker_block_idx % (reduce_add_level * 2) == 0 && (in_data_set_idx + FSV * reduce_add_level) < num_local_workers) {
            var_per_feature[in_data_set_idx] += var_per_feature[in_data_set_idx + FSV * reduce_add_level];
        }
        reduce_add_level *= 2;
    }

    if (worker_block_idx == 0 && (f_base + get_sub_group_local_id()) < INPUT0_FEATURE_NUM) {
        variance = var_per_feature[in_data_set_idx] / TO_ACCUMULATOR_TYPE(data_set_size);
        internal_variance[bf] = variance;
    }
}
#elif GROUP_NORM_KERNEL_GROUP_VAR
KERNEL(calc_var_per_group)(
    __global ACCUMULATOR_TYPE* internal_variance
) {
    const uint data_idx = get_global_id(0) + get_global_id(1) * GWS0;
    const uint num_workers = LWS0;
    const uint group_size = GWS0 / NUM_GROUPS;
    const uint items_num = group_size / num_workers;

    if ((data_idx % group_size) < num_workers) {
        ACCUMULATOR_TYPE my_variance = ACCUMULATOR_VAL_ZERO;
        for (uint i = 0; i < items_num; ++i) {
            my_variance += internal_variance[data_idx + num_workers * i];
        }

        ACCUMULATOR_TYPE variance = work_group_reduce_add(my_variance);
        variance /= TO_ACCUMULATOR_TYPE(group_size);
        variance = native_powr(variance + TO_ACCUMULATOR_TYPE(EPSILON), -0.5f);
        for (uint i = 0; i < items_num; ++i) {
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
    const uint f = get_global_id(1) / OUTPUT_BATCH_NUM * FSV + get_sub_group_local_id();
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
            output[output_index] = TO_OUTPUT_TYPE(ACTIVATION(normalized, ACTIVATION_PARAMS));
        #endif
    } else {
        output[output_index] = OUTPUT_VAL_ZERO;
    }
}
#endif
