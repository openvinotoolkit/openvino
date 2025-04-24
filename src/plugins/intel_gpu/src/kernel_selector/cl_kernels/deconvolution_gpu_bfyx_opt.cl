// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define WORK_GROUP_GROUP_SIZE 16

__attribute__((reqd_work_group_size(WORK_GROUP_GROUP_SIZE, 1, 1)))
KERNEL(deconvolution_gpu_bfyx_opt)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* filter
#if BIAS_TERM
    , const __global BIAS_TYPE* bias
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;

    const uint b_f          = get_global_id(2);
    const uint batch_offset = b_f / OUTPUT_FEATURE_NUM;
    const uint ofm_offset   = b_f % OUTPUT_FEATURE_NUM;

#if Y_AXIS_1D_FILTER == 1
    const uint global_y_group    = get_group_id(0);
    const uint global_x_group    = get_group_id(1);

    const uint local_y        = get_local_id(0);
    const uint local_x        = get_local_id(1);

    const uint stride_y_id = global_y_group % STRIDE_SIZE_Y;
    const uint stride_x_id = global_x_group % STRIDE_SIZE_X;

    const uint id_y = (global_y_group / STRIDE_SIZE_Y) * STRIDE_SIZE_Y * WORK_GROUP_GROUP_SIZE + local_y * STRIDE_SIZE_Y + stride_y_id;

    if (id_y >= OUTPUT_SIZE_Y)
        return;

    const uint id_x = (global_x_group / STRIDE_SIZE_X) * STRIDE_SIZE_X + local_x * STRIDE_SIZE_X + stride_x_id;
#else // Y_AXIS_1D_FILTER == 1
    const uint global_x_group    = get_group_id(0);
    const uint global_y_group    = get_group_id(1);

    const uint local_x        = get_local_id(0);
    const uint local_y        = get_local_id(1);

    const uint stride_x_id = global_x_group % STRIDE_SIZE_X;
    const uint stride_y_id = global_y_group % STRIDE_SIZE_Y;

    const uint id_x = (global_x_group / STRIDE_SIZE_X) * STRIDE_SIZE_X * WORK_GROUP_GROUP_SIZE + local_x * STRIDE_SIZE_X + stride_x_id;

    if (id_x >= OUTPUT_SIZE_X)
        return;

    const uint id_y = (global_y_group / STRIDE_SIZE_Y) * STRIDE_SIZE_Y + local_y * STRIDE_SIZE_Y + stride_y_id;
#endif // Y_AXIS_1D_FILTER == 1

    const int in_x = (int)id_x + PADDING_SIZE_X - (FILTER_SIZE_X - 1);
    const int in_y = (int)id_y + PADDING_SIZE_Y - (FILTER_SIZE_Y - 1);

    const uint start_x = (STRIDE_SIZE_X - (in_x % STRIDE_SIZE_X)) % STRIDE_SIZE_X;
    const uint start_y = (STRIDE_SIZE_Y - (in_y % STRIDE_SIZE_Y)) % STRIDE_SIZE_Y;

#if GROUPED
    const uint g = (ofm_offset / FILTER_OFM_NUM);
    const uint of = (ofm_offset % FILTER_OFM_NUM);
    const uint in_split_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    const uint filter_offset = g * FILTER_GROUPS_PITCH;
#else
    const uint g = 0;
    const uint of = ofm_offset;
    const uint in_split_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    const uint filter_offset = 0;
#endif

    const uint input_offset = INPUT0_OFFSET + batch_offset*INPUT0_BATCH_PITCH + in_split_offset;

#if Y_AXIS_1D_FILTER == 1
    for (uint i = start_x; i < FILTER_SIZE_X; i+=STRIDE_SIZE_X)
    {
        const int input_offset_x = in_x + i;
        const bool zero_x = (input_offset_x >= INPUT0_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0);

        if(!zero_x)
        {
            for (uint j = start_y; j < FILTER_SIZE_Y; j+=STRIDE_SIZE_Y)
            {
                const int input_offset_y = in_y + j;
                const bool zero_y = (input_offset_y >= INPUT0_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0);

                if(!zero_y)
#else // Y_AXIS_1D_FILTER == 1
    for (uint i = start_y; i < FILTER_SIZE_Y; i+=STRIDE_SIZE_Y)
    {
        const int input_offset_y = in_y + i;
        const bool zero_y = (input_offset_y >= INPUT0_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0);

        if(!zero_y)
        {
            for (uint j = start_x; j < FILTER_SIZE_X; j+=STRIDE_SIZE_X)
            {
                const int input_offset_x = in_x + j;
                const bool zero_x = (input_offset_x >= INPUT0_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0);

                if(!zero_x)
#endif // Y_AXIS_1D_FILTER == 1
                {
                    uint fixed_input_offset_x = (uint)input_offset_x / STRIDE_SIZE_X;
                    uint fixed_input_offset_y = (uint)input_offset_y / STRIDE_SIZE_Y;
                    uint input_idx = input_offset + (uint)fixed_input_offset_x*INPUT0_X_PITCH + (uint)fixed_input_offset_y*INPUT0_Y_PITCH;

                    uint filter_idx = filter_offset + of*FILTER_OFM_PITCH + (FILTER_SIZE_Y - i - 1)*FILTER_Y_PITCH + (FILTER_SIZE_X - j - 1)*FILTER_X_PITCH;
                    for (uint h = 0; h < FILTER_IFM_NUM; h++)
                    {
                        acc += TO_ACCUMULATOR_TYPE(input[input_idx]) * TO_ACCUMULATOR_TYPE(filter[filter_idx]);
                        filter_idx += FILTER_IFM_PITCH;
                        input_idx += INPUT0_FEATURE_PITCH;
                    }
                }
            }
        }
    }

    ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(acc);
#if BIAS_TERM
    result += bias[ofm_offset];
#endif
    result = ACTIVATION(result, ACTIVATION_PARAMS);

    const uint out_split_offset = g * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    const uint dst_index = OUTPUT_OFFSET + out_split_offset + batch_offset*OUTPUT_BATCH_PITCH + of*OUTPUT_FEATURE_PITCH + id_y*OUTPUT_Y_PITCH + id_x*OUTPUT_X_PITCH;

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[dst_index] = FUSED_OPS_RESULT;
#else
    output[dst_index] = TO_OUTPUT_TYPE(result);
#endif
}

#undef ACTIVATION
#undef WORK_GROUP_GROUP_SIZE
