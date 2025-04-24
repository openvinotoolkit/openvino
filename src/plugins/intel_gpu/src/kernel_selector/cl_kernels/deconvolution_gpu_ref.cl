// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(deconvolution_gpu_yxfb_ref)(
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

#if DIM_ORDER_XYBF == 1
    const uint out_x        = get_global_id(0);
#if  OUTPUT_SIZE_Z == 1
    const uint out_y        = get_global_id(1);
    const uint out_z        = 0;
#else // 3D
    const uint out_y        = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
    const uint out_z        = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
#endif // 2D/3D
    const uint b_f          = get_global_id(2);
    const uint batch_offset = b_f / OUTPUT_FEATURE_NUM;
    const uint ofm_offset   = b_f % OUTPUT_FEATURE_NUM;

    if (out_x >= OUTPUT_SIZE_X)
        return;
#else
    const uint b_f           = get_global_id(0);
    const uint out_x         = (uint)get_global_id(1);
#if  OUTPUT_SIZE_Z == 1
    const uint out_y         = (uint)get_global_id(2);
    const uint out_z        = 0;
#else // 3D
    const uint out_y        = (uint)get_global_id(2) % OUTPUT_SIZE_Y;
    const uint out_z        = (uint)get_global_id(2) / OUTPUT_SIZE_Y;
#endif // 2D/3D
    const uint ofm_offset    = b_f / INPUT0_BATCH_NUM;
    const uint batch_offset  = b_f % INPUT0_BATCH_NUM;
#endif

    const int x = (int)out_x + PADDING_SIZE_X - (FILTER_SIZE_X - 1);
    const int y = (int)out_y + PADDING_SIZE_Y - (FILTER_SIZE_Y - 1);
    const int z = (int)out_z + PADDING_SIZE_Z - (FILTER_SIZE_Z - 1);

#if GROUPED
    const uint g = (ofm_offset / FILTER_OFM_NUM);
    const uint of = (ofm_offset % FILTER_OFM_NUM);
    const uint filter_offset = g * FILTER_GROUPS_PITCH;
#else
    const uint g = 0;
    const uint of = ofm_offset;
    const uint filter_offset = 0;
#endif

    const uint in_split_offset = g * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
    const uint input_offset = INPUT0_OFFSET + batch_offset*INPUT0_BATCH_PITCH + in_split_offset;

    for (uint k = 0; k < FILTER_SIZE_Z; k++)
    {
        const int input_offset_z = z + k;
        const bool zero_z = (input_offset_z >= INPUT0_SIZE_Z * STRIDE_SIZE_Z) || (input_offset_z < 0) || ((input_offset_z % STRIDE_SIZE_Z) != 0);

        if(!zero_z)
        {
            for (uint i = 0; i < FILTER_SIZE_Y; i++)
            {
                const int input_offset_y = y + i;
                const bool zero_y = (input_offset_y >= INPUT0_SIZE_Y * STRIDE_SIZE_Y) || (input_offset_y < 0) || ((input_offset_y % STRIDE_SIZE_Y) != 0);

                if(!zero_y)
                {
                    for (uint j = 0; j < FILTER_SIZE_X; j++)
                    {
                        const int input_offset_x = x + j;
                        const bool zero_x = (input_offset_x >= INPUT0_SIZE_X * STRIDE_SIZE_X) || (input_offset_x < 0) || ((input_offset_x % STRIDE_SIZE_X) != 0);

                        if(!zero_x)
                        {
                            uint fixed_input_offset_x = (uint)input_offset_x / STRIDE_SIZE_X;
                            uint fixed_input_offset_y = (uint)input_offset_y / STRIDE_SIZE_Y;
                            uint fixed_input_offset_z = (uint)input_offset_z / STRIDE_SIZE_Z;

                            uint input_idx;
#if INPUT0_SIMPLE
                            input_idx = input_offset + (uint)fixed_input_offset_x*INPUT0_X_PITCH + (uint)fixed_input_offset_y*INPUT0_Y_PITCH + (uint)fixed_input_offset_z*INPUT0_Z_PITCH;
#endif

                            uint filter_idx = filter_offset + of*FILTER_OFM_PITCH + (FILTER_SIZE_Z - k - 1)*FILTER_Z_PITCH + (FILTER_SIZE_Y - i - 1)*FILTER_Y_PITCH + (FILTER_SIZE_X - j - 1)*FILTER_X_PITCH;
                            for (uint h = 0; h < FILTER_IFM_NUM; h++) {
#if !INPUT0_SIMPLE
#   if INPUT0_DIMS <= 4
                                input_idx = INPUT0_GET_INDEX(batch_offset, h + g*FILTER_IFM_NUM, fixed_input_offset_y, fixed_input_offset_x);
#   elif INPUT0_DIMS == 5
                                input_idx = INPUT0_GET_INDEX(batch_offset, h + g*FILTER_IFM_NUM, fixed_input_offset_z, fixed_input_offset_y, fixed_input_offset_x);
#   endif
#endif

                                acc += TO_ACCUMULATOR_TYPE(input[input_idx]) * TO_ACCUMULATOR_TYPE(filter[filter_idx]);
                                filter_idx += FILTER_IFM_PITCH;
#if INPUT0_SIMPLE
                                input_idx += INPUT0_FEATURE_PITCH;
#endif
                            }
                        }
                    }
                }
            }
        }
    }

    ACTIVATION_TYPE pre_activation = TO_ACTIVATION_TYPE(acc);
#if BIAS_TERM
    pre_activation += TO_ACTIVATION_TYPE(bias[ofm_offset]);
#endif
    ACTIVATION_TYPE post_activation = ACTIVATION(pre_activation, ACTIVATION_PARAMS);

    OUTPUT_TYPE result;
#if HAS_FUSED_OPS
    FUSED_OPS;
    result = FUSED_OPS_RESULT;
#else
    result = TO_OUTPUT_TYPE(post_activation);
#endif

#if OUTPUT_DIMS <= 4
    const uint dst_index = OUTPUT_GET_INDEX(batch_offset, g * FILTER_OFM_NUM + of, out_y, out_x);
#elif OUTPUT_DIMS == 5
    const uint dst_index = OUTPUT_GET_INDEX(batch_offset, g * FILTER_OFM_NUM + of, out_z, out_y, out_x);
#else
#   error deconvolution_gpu_ref.cl - Unsupported number of output dimensions.
#endif

    output[dst_index] = result;
}
