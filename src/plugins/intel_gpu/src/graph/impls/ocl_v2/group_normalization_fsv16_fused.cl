// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#ifdef GROUP_NORM_KERNEL_FEATURE_MEAN_SQR_MEAN
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(calc_mean_sqr_mean_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* scale,
    const __global INPUT2_TYPE* bias,
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    __global OUTPUT_TYPE* restrict output
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
    const uint input_data_offset = data_set_offset + in_data_set_idx;

    __local ACCUMULATOR_TYPE sum_per_feature[SLM_SIZE];
    __local ACCUMULATOR_TYPE sqr_sum_per_feature[SLM_SIZE];

    ACCUMULATOR_TYPE sum = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE sqr_sum = ACCUMULATOR_VAL_ZERO;

    for (uint i = 0; i < items_num; ++i) {
        ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[input_data_offset + i * workers_per_dataset * FSV]);
        sum += data;
        sqr_sum += data * data;
    }

    if (in_data_set_idx < leftovers) {
        ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[input_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx]);
        sum += data;
        sqr_sum += data * data;
    }

    sum_per_feature[in_data_set_idx] = sum;
    sqr_sum_per_feature[in_data_set_idx] = sqr_sum;
    const uint num_local_workers = LWS0;
    const uint worker_block_idx = in_data_set_idx / FSV;
    uint reduce_add_level = 1;
    while ((SLM_SIZE / FSV) > reduce_add_level) {
        barrier(CLK_LOCAL_MEM_FENCE);
        if (worker_block_idx % (reduce_add_level * 2) == 0 && (in_data_set_idx + FSV * reduce_add_level) < num_local_workers) {
            sum_per_feature[in_data_set_idx] += sum_per_feature[in_data_set_idx + FSV * reduce_add_level];
            sqr_sum_per_feature[in_data_set_idx] += sqr_sum_per_feature[in_data_set_idx + FSV * reduce_add_level];
        }
        reduce_add_level *= 2;
    }
    
    // at this point, block 0 has fully reduced values. divide by xy size
    ACCUMULATOR_TYPE mean = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE variance = ACCUMULATOR_VAL_ZERO;
    if (worker_block_idx == 0 && (f_base + in_data_set_idx) < INPUT0_FEATURE_NUM) {
        mean = sum_per_feature[in_data_set_idx] / TO_ACCUMULATOR_TYPE(data_set_size);
        variance = sqr_sum_per_feature[in_data_set_idx] / TO_ACCUMULATOR_TYPE(data_set_size);
    }
    barrier(0);

    // the sums are only meaningful in block 0, but all will read from there
    ACCUMULATOR_TYPE sum_mean = work_group_scan_inclusive_add(mean);
    ACCUMULATOR_TYPE sum_variance = work_group_scan_inclusive_add(variance);
    barrier(0);
    mean = ACCUMULATOR_VAL_ZERO;
    variance = ACCUMULATOR_VAL_ZERO;
    const uint group_size = INPUT0_FEATURE_NUM / NUM_GROUPS;
    const uint groups_in_fsv = FSV / group_size;
    for (uint i = 0; i < groups_in_fsv; ++i) {
        // take the sum from the end of a group in block 0
        ACCUMULATOR_TYPE group_mean = work_group_broadcast(sum_mean, (i + 1) * group_size - 1);
        ACCUMULATOR_TYPE group_variance = work_group_broadcast(sum_variance, (i + 1) * group_size - 1);
        if ((in_data_set_idx % FSV) / group_size == i + 1) {
            // if previous group, save
            mean = group_mean;
            variance = group_variance;
        } else if ((in_data_set_idx % FSV) / group_size == i) {
            // if my group, subtract sum up to prior group to get final value
            mean = group_mean - mean;
            variance = group_variance - variance;
        }
    }
    // at this stage, every WI has a correct sum loaded from block 0
    mean /= TO_ACCUMULATOR_TYPE(group_size);
    variance /= TO_ACCUMULATOR_TYPE(group_size);
    variance -= mean * mean;
    variance = native_powr(variance + TO_ACCUMULATOR_TYPE(EPSILON), -0.5f);

    const uint f = f_base + in_data_set_idx % FSV;
    const uint output_base_offset = OUTPUT_GET_INDEX(b, f_base, 0, 0);
    const uint output_data_offset = output_base_offset + in_data_set_idx;
    ACTIVATION_TYPE scale_f = TO_ACTIVATION_TYPE(scale[f]);
    ACTIVATION_TYPE bias_f = TO_ACTIVATION_TYPE(bias[f]);
    #define CHUNK_SIZE 8
    ACTIVATION_TYPE input_data[CHUNK_SIZE];

    for (uint j = 0; j < (items_num + CHUNK_SIZE - 1) / CHUNK_SIZE; ++j) {
        for (uint i = 0; (i < CHUNK_SIZE) && (i + j * CHUNK_SIZE < items_num); ++i) {
            input_data[i] = TO_ACTIVATION_TYPE(input[input_data_offset + (i + j * CHUNK_SIZE) * workers_per_dataset * FSV]);
            input_data[i] = (input_data[i] - mean) * variance;
        }

        for (uint i = 0; (i < CHUNK_SIZE) && (i + j * CHUNK_SIZE < items_num); ++i) {
            ACTIVATION_TYPE normalized = input_data[i] * scale_f + bias_f;
            if (f < OUTPUT_FEATURE_NUM) {
                #if HAS_FUSED_OPS
                    FUSED_OPS;
                    output[output_data_offset + (i  + j * CHUNK_SIZE) * workers_per_dataset * FSV] = FUSED_OPS_RESULT;
                #else
                    output[output_data_offset + (i  + j * CHUNK_SIZE) * workers_per_dataset * FSV] = TO_OUTPUT_TYPE(normalized);
                #endif
            } else {
                output[output_data_offset + (i  + j * CHUNK_SIZE) * workers_per_dataset * FSV] = OUTPUT_VAL_ZERO;
            }
        }
    }

    if (in_data_set_idx < leftovers) {
        ACTIVATION_TYPE normalized = (TO_ACTIVATION_TYPE(input[input_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx]) - mean) * variance;
        normalized = normalized * scale_f + bias_f;
        if (f < OUTPUT_FEATURE_NUM) {
            #if HAS_FUSED_OPS
                FUSED_OPS;
                output[output_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx] = FUSED_OPS_RESULT;
            #else
                output[output_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx] = TO_OUTPUT_TYPE(normalized);
            #endif
        } else {
            output[output_data_offset + items_num * workers_per_dataset * FSV + in_data_set_idx] = OUTPUT_VAL_ZERO;
        }
    }
}
#endif
