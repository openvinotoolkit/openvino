// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"

#ifdef GROUP_NORM_KERNEL_FEATURE_MEAN_SQR_MEAN
#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
#endif
KERNEL(calc_mean_sqr_mean_per_feature)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global ACCUMULATOR_TYPE* internal_mean,
    __global ACCUMULATOR_TYPE* internal_sqr_mean
) {
    #if INPUT0_DIMS == 5
        const uint bf = get_global_id(2) / LWS2;     // batch * feature
    #else
        const uint bf = get_global_id(2);     // batch * feature
    #endif
    const uint b = bf / INPUT0_FEATURE_NUM;
    const uint f = bf % INPUT0_FEATURE_NUM;
    #if INPUT0_DIMS == 5
        const uint z_num_workers = LWS2;
    #endif
    const uint y_num_workers = LWS1;
    const uint x_num_workers = LWS0;
    #if INPUT0_DIMS == 5
        const uint z_block_size = INPUT0_SIZE_Z / z_num_workers;
        const uint z_base = get_local_id(2) * z_block_size;
        const uint z_leftover = INPUT0_SIZE_Z - z_num_workers * z_block_size;
    #endif
    const uint y_block_size = INPUT0_SIZE_Y / y_num_workers;
    const uint y_base = get_local_id(1) * y_block_size;
    const uint y_leftover = INPUT0_SIZE_Y - y_num_workers * y_block_size;

    const uint x_block_size = INPUT0_SIZE_X / x_num_workers;
    const uint x_base = get_local_id(0);
    const uint x_leftover = INPUT0_SIZE_X - x_num_workers * x_block_size;

    ACCUMULATOR_TYPE mean = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE sqr_mean = ACCUMULATOR_VAL_ZERO;

    #if INPUT0_DIMS == 5
        for (uint z = z_base; z < (z_base + z_block_size); ++z) {
    #endif
            for (uint y = y_base; y < (y_base + y_block_size); ++y) {
                #if INPUT0_DIMS == 5
                    uint my_data_offset = INPUT0_GET_INDEX(b, f, z, y, x_base);
                #else
                    uint my_data_offset = INPUT0_GET_INDEX(b, f, y, x_base);
                #endif
                for (uint i = 0; i < x_block_size; ++i) {
                    ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]);
                    mean += data;
                    sqr_mean += data * data;
                }
            }
    #if INPUT0_DIMS == 5
        }
    #endif

    #if INPUT0_DIMS == 5
        if (get_local_id(2) < z_leftover) {
            for (uint y = y_base; y < (y_base + y_block_size); ++y) {
                uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(2) + z_num_workers * z_block_size), y, x_base);
                for (uint i = 0; i < x_block_size; ++i) {
                    ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]);
                    mean += data;
                    sqr_mean  += data * data;
                }
            }
        }
    #endif

    if (get_local_id(1) < y_leftover) {
        #if INPUT0_DIMS == 5
            for (uint z = z_base; z < (z_base + z_block_size); ++z) {
                uint my_data_offset = INPUT0_GET_INDEX(b, f, z, (get_local_id(1) + y_num_workers * y_block_size), x_base);
        #else
                uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size), x_base);
        #endif
                for (uint i = 0; i < x_block_size; ++i) {
                    ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[my_data_offset + i * x_num_workers]);
                    mean += data;
                    sqr_mean += data * data;
                }
        #if INPUT0_DIMS == 5
            }
        #endif
    }

    if (get_local_id(0) < x_leftover) {
        #if INPUT0_DIMS == 5
            for (uint z = z_base; z < (z_base + z_block_size); ++z) {
        #endif
                for (uint y = y_base; y < (y_base + y_block_size); ++y) {
                    #if INPUT0_DIMS == 5
                        uint my_data_offset = INPUT0_GET_INDEX(b, f, z, y, (get_local_id(0) + x_num_workers * x_block_size));
                    #else
                        uint my_data_offset = INPUT0_GET_INDEX(b, f, y, (get_local_id(0) + x_num_workers * x_block_size));
                    #endif
                    ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[my_data_offset]);
                    mean += data;
                    sqr_mean += data * data;
                }
        #if INPUT0_DIMS == 5
            }
        #endif
    }

    #if INPUT0_DIMS == 5
        if (get_local_id(2) < z_leftover && get_local_id(1) < y_leftover && get_local_id(0) < x_leftover) {
            uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(2) + z_num_workers * z_block_size),
                                                         (get_local_id(1) + y_num_workers * y_block_size),
                                                         (get_local_id(0) + x_num_workers * x_block_size));
    #else
        if (get_local_id(1) < y_leftover && get_local_id(0) < x_leftover) {
            uint my_data_offset = INPUT0_GET_INDEX(b, f, (get_local_id(1) + y_num_workers * y_block_size),
                                                         (get_local_id(0) + x_num_workers * x_block_size));
    #endif
            ACCUMULATOR_TYPE data = TO_ACCUMULATOR_TYPE(input[my_data_offset]);
            mean += data;
            sqr_mean += data * data;
        }

    #if INPUT0_DIMS == 5
        const uint num_local_workers = z_num_workers * y_num_workers * x_num_workers;
    #else
        const uint num_local_workers = y_num_workers * x_num_workers;
    #endif
    const uint worker_idx = get_local_linear_id();

    mean = work_group_reduce_add(mean);
    sqr_mean = work_group_reduce_add(sqr_mean);

    if (worker_idx == 0) {
        mean = mean / TO_ACCUMULATOR_TYPE(INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X);
        sqr_mean = sqr_mean / TO_ACCUMULATOR_TYPE(INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X);
        internal_mean[bf] = mean;
        internal_sqr_mean[bf] = sqr_mean;
    }
}
#elif GROUP_NORM_KERNEL_GROUP_MEAN_VARIANCE
#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
#endif
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
#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS0, LWS1, LWS2)))
#endif
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
    const uint bf = get_global_id(1);
    const uint b = bf / OUTPUT_FEATURE_NUM;
    const uint f = bf % OUTPUT_FEATURE_NUM;
    #if INPUT0_DIMS == 5
        const uint zyx = get_global_id(0);
        const uint z = zyx / (OUTPUT_SIZE_Y * OUTPUT_SIZE_X);
        const uint yx = zyx % (OUTPUT_SIZE_Y * OUTPUT_SIZE_X);
    #else
        const uint yx = get_global_id(0);
    #endif
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;

    #if INPUT0_DIMS == 5
        const uint input_data_index = INPUT0_GET_INDEX(b, f, z, y, x);
    #else
        const uint input_data_index = INPUT0_GET_INDEX(b, f, y, x);
    #endif

    ACTIVATION_TYPE mean = TO_ACTIVATION_TYPE(internal_mean[bf]);
    ACTIVATION_TYPE variance = TO_ACTIVATION_TYPE(internal_variance[bf]);
    ACTIVATION_TYPE normalized = (TO_ACTIVATION_TYPE(input[input_data_index]) - mean) * variance;
    normalized = normalized * TO_ACTIVATION_TYPE(scale[f]) + TO_ACTIVATION_TYPE(bias[f]);

    #if INPUT0_DIMS == 5
        const uint output_data_index = OUTPUT_GET_INDEX(b, f, z, y, x);
    #else
        const uint output_data_index = OUTPUT_GET_INDEX(b, f, y, x);
    #endif
    #if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_data_index] = FUSED_OPS_RESULT;
    #else
        output[output_data_index] = TO_OUTPUT_TYPE(normalized);
    #endif
}
#endif
