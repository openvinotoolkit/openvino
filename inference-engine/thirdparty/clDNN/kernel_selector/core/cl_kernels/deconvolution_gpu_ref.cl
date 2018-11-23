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
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* filter,
#if BIAS_TERM
    const __global UNIT_TYPE* bias,
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
    const uint out_y        = get_global_id(1);
    const uint b_f          = get_global_id(2);
    const uint batch_offset = b_f / OUTPUT_FEATURE_NUM;
    const uint ofm_offset   = b_f % OUTPUT_FEATURE_NUM;
    
    if (out_x >= OUTPUT_SIZE_X)
        return;
#else
    const uint b_f           = get_global_id(0);
    const uint out_x         = (uint)get_global_id(1);
    const uint out_y         = (uint)get_global_id(2);
    const uint ofm_offset    = b_f / INPUT0_BATCH_NUM;
    const uint batch_offset  = b_f % INPUT0_BATCH_NUM;
#endif 

    const int x = (int)out_x + PADDING_SIZE_X - (FILTER_SIZE_X - 1);
    const int y = (int)out_y + PADDING_SIZE_Y - (FILTER_SIZE_Y - 1);
    
#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (ofm_offset / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    const uint input_offset = INPUT0_OFFSET + batch_offset*INPUT0_BATCH_PITCH + in_split_offset;

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
                    uint input_idx = input_offset + (uint)fixed_input_offset_x*INPUT0_X_PITCH + (uint)fixed_input_offset_y*INPUT0_Y_PITCH;
#if GRADIENT
                    uint filter_idx = ofm_offset*FILTER_IFM_PITCH + (FILTER_SIZE_Y - i - 1)*FILTER_Y_PITCH + (FILTER_SIZE_X - j - 1)*FILTER_X_PITCH;
                    for (uint h = 0; h < FILTER_OFM_NUM; h++)
                    {
                        result = fma(input[input_idx], filter[filter_idx], result);
                        filter_idx += FILTER_OFM_PITCH;
                        input_idx += INPUT0_FEATURE_PITCH;
                    }
#else
                    uint filter_idx = ofm_offset*FILTER_OFM_PITCH + (FILTER_SIZE_Y - i - 1)*FILTER_Y_PITCH + (FILTER_SIZE_X - j - 1)*FILTER_X_PITCH;
                    for (uint h = 0; h < FILTER_IFM_NUM; h++)
                    {
                        result = fma(input[input_idx], filter[filter_idx], result);
                        filter_idx += FILTER_IFM_PITCH;
                        input_idx += INPUT0_FEATURE_PITCH;
                    }
#endif
                }
            }
        }
    }
#if BIAS_TERM
    result += bias[ofm_offset];
#endif
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * FILTER_OFM_NUM;
    const uint dst_index = OUTPUT_OFFSET + out_split_offset + batch_offset*OUTPUT_BATCH_PITCH + ofm_offset*OUTPUT_FEATURE_PITCH + out_y*OUTPUT_Y_PITCH + out_x*OUTPUT_X_PITCH;
#if FUSED_ELTWISE
    const uint fused_index = INPUT1_OFFSET + split_idx * INPUT1_FEATURE_PITCH * FILTER_OFM_NUM + batch_offset*INPUT1_BATCH_PITCH + ofm_offset*INPUT1_FEATURE_PITCH + out_y*INPUT1_Y_PITCH + out_x*INPUT1_X_PITCH;
#if !GRADIENT
	output[dst_index] = ACTIVATION(result + fuse_input[fused_index], NL_M, NL_N);
#else
	output[dst_index] = result + fuse_input[fused_index];
#endif

#else
    output[dst_index] = ACTIVATION(result, NL_M, NL_N);
#endif
}

#undef ACTIVATION
