// Copyright (c) 2016-2017 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "include/include_all.cl"

KERNEL(deconvolution_gpu_yxfb_ref)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* filter,
#if BIAS_TERM
    const __global BIAS_TYPE* bias,
#endif
    uint split_idx
#if FUSED_ELTWISE
	, const __global UNIT_TYPE* fuse_input
#endif
	)
{
    UNIT_TYPE result = UNIT_VAL_ZERO;

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

#if GROUPED || DEPTHWISE_SEPARABLE_OPT
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
#if OUTPUT_LAYOUT_B_FS_ZYX_FSV16 || OUTPUT_LAYOUT_B_FS_YX_FSV16 || OUTPUT_LAYOUT_BS_FS_ZYX_BSV16_FSV16
                            uint input_idx;
#else
                            uint input_idx = input_offset + (uint)fixed_input_offset_x*INPUT0_X_PITCH + (uint)fixed_input_offset_y*INPUT0_Y_PITCH + (uint)fixed_input_offset_z*INPUT0_Z_PITCH;
#endif
#if GRADIENT
                            uint filter_idx = filter_offset + of*FILTER_IFM_PITCH + (FILTER_SIZE_Z - k - 1)*FILTER_Z_PITCH + (FILTER_SIZE_Y - i - 1)*FILTER_Y_PITCH + (FILTER_SIZE_X - j - 1)*FILTER_X_PITCH;
                            for (uint h = 0; h < FILTER_OFM_NUM; h++)
                            {
#if INPUT0_LAYOUT_B_FS_ZYX_FSV16 || INPUT0_LAYOUT_BS_FS_ZYX_BSV16_FSV16
                                input_idx = INPUT0_GET_INDEX(batch_offset, h + g*FILTER_IFM_NUM, fixed_input_offset_z, fixed_input_offset_y, fixed_input_offset_x);
#elif INPUT0_LAYOUT_BS_FS_YX_FSV16
                                input_idx = INPUT0_GET_INDEX(batch_offset, h + g*FILTER_IFM_NUM, fixed_input_offset_y, fixed_input_offset_x);
#endif
                                result = fma(input[input_idx], filter[filter_idx], result);
                                filter_idx += FILTER_OFM_PITCH;
#if !INPUT0_LAYOUT_B_FS_ZYX_FSV16 && !INPUT0_LAYOUT_BS_FS_ZYX_BSV16_FSV16 && !INPUT0_LAYOUT_B_FS_YX_FSV16
                                input_idx += INPUT0_FEATURE_PITCH;
#endif
                            }
#else
                            uint filter_idx = filter_offset + of*FILTER_OFM_PITCH + (FILTER_SIZE_Z - k - 1)*FILTER_Z_PITCH + (FILTER_SIZE_Y - i - 1)*FILTER_Y_PITCH + (FILTER_SIZE_X - j - 1)*FILTER_X_PITCH;
                            for (uint h = 0; h < FILTER_IFM_NUM; h++)
                            {
#if OUTPUT_LAYOUT_B_FS_ZYX_FSV16 || OUTPUT_LAYOUT_BS_FS_ZYX_BSV16_FSV16
                                input_idx = INPUT0_GET_INDEX(batch_offset, h + g*FILTER_IFM_NUM, fixed_input_offset_z, fixed_input_offset_y, fixed_input_offset_x);
#elif OUTPUT_LAYOUT_B_FS_YX_FSV16
                                input_idx = INPUT0_GET_INDEX(batch_offset, h + g*FILTER_IFM_NUM, fixed_input_offset_y, fixed_input_offset_x);
#endif
                                result = fma(input[input_idx], filter[filter_idx], result);
                                filter_idx += FILTER_IFM_PITCH;
#if !OUTPUT_LAYOUT_B_FS_ZYX_FSV16 && !OUTPUT_LAYOUT_B_FS_YX_FSV16 && !OUTPUT_LAYOUT_BS_FS_ZYX_BSV16_FSV16
                                input_idx += INPUT0_FEATURE_PITCH;
#endif
                            }
#endif
                        }
                    }
                }
            }
        }
    }

#if BIAS_TERM
    result += bias[ofm_offset];
#endif
    const uint out_split_offset = g * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
#if OUTPUT_LAYOUT_B_FS_ZYX_FSV16 || OUTPUT_LAYOUT_BS_FS_ZYX_BSV16_FSV16
    const uint dst_index = OUTPUT_OFFSET + OUTPUT_GET_INDEX(batch_offset, g * FILTER_OFM_NUM + of, out_z, out_y, out_x);
#elif OUTPUT_LAYOUT_B_FS_YX_FSV16
    const uint dst_index = OUTPUT_OFFSET + OUTPUT_GET_INDEX(batch_offset, g * FILTER_OFM_NUM + of, out_y, out_x);
#else
    const uint dst_index = OUTPUT_OFFSET + out_split_offset + batch_offset*OUTPUT_BATCH_PITCH + of*OUTPUT_FEATURE_PITCH + out_z*OUTPUT_Z_PITCH + out_y*OUTPUT_Y_PITCH + out_x*OUTPUT_X_PITCH;
#endif
#if FUSED_ELTWISE
#if OUTPUT_LAYOUT_B_FS_ZYX_FSV16 || OUTPUT_LAYOUT_BS_FS_ZYX_BSV16_FSV16
    const uint fused_index = INPUT1_OFFSET + INPUT1_GET_INDEX(batch_offset, g * FILTER_OFM_NUM + of, out_z, out_y, out_x);
#elif OUTPUT_LAYOUT_B_FS_YX_FSV16
    const uint fused_index = INPUT1_OFFSET + INPUT1_GET_INDEX(batch_offset, g * FILTER_OFM_NUM + of, out_y, out_x);
#else
    const uint fused_index = INPUT1_OFFSET + split_idx * INPUT1_FEATURE_PITCH * FILTER_OFM_NUM + batch_offset*INPUT1_BATCH_PITCH + of*INPUT1_FEATURE_PITCH + out_z*INPUT1_Z_PITCH + out_y*INPUT1_Y_PITCH + out_x*INPUT1_X_PITCH;
#endif
#if !GRADIENT
	output[dst_index] = ACTIVATION(result + fuse_input[fused_index], ACTIVATION_PARAMS);
#else
	output[dst_index] = result + fuse_input[fused_index];
#endif

#else
    output[dst_index] = ACTIVATION(result, ACTIVATION_PARAMS);
#endif

}

#undef ACTIVATION
