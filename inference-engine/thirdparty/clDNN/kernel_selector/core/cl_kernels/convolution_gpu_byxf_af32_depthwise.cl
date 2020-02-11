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

#include "include/common.cl"

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

KERNEL(convolution_gpu_byxf_af32_depthwise)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    uint split_idx)
{
    const uint x = get_global_id(1);
    const uint y = get_global_id(2);
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(0);
    const uint b = 0;
#else
    const uint f = (uint)get_global_id(0) % OUTPUT_FEATURE_NUM;
    const uint b = (uint)get_global_id(0) / OUTPUT_FEATURE_NUM;
#endif

    int dotProd = 0;
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

#if DEPTHWISE_SEPARABLE_OPT
    const uint g = (f / FILTER_OFM_NUM);
    const uint of = f;
#else
    const uint g = split_idx;
    const uint of = f + split_idx*FILTER_OFM_NUM;
#endif

    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + g*FILTER_IFM_NUM;

    for (uint k = 0; k < FILTER_IFM_NUM; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;
            const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

            if(!zero_y)
            {
                for (uint i = 0; i < FILTER_SIZE_X ; ++i)
                {
                    const int input_offset_x = input_x + i * DILATION_SIZE_X;
                    const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                    if(!zero_x)
                    {
                        uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH + k*INPUT0_FEATURE_PITCH;
                        uint filter_idx = filter_offset + k*FILTER_IFM_PITCH + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;
                        dotProd += (int)input[input_idx] * (int)weights[filter_idx];
                    }
                }
            }
        }
    }

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    #if OUTPUT_LAYOUT_BYXF_AF32 == 1
        const uint bias_index = GET_DATA_INDEX(BIAS, b, of, y, x);
    #elif OUTPUT_LAYOUT_B_FS_YX_FSV4 == 1
        const uint bias_index = GET_DATA_B_FS_YX_FSV4_INDEX(BIAS, b, of, y, x);
    #else
        #error "Incorrect output layout"
    #endif
#elif BIAS_PER_OFM
    const uint bias_index = of;
#endif

     // TODO: Maybe half should be supported as well.
     float res = (float)dotProd + biases[bias_index];
#else
     float res = (float)dotProd;
#endif
#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE out = FINAL_NAME;
#else
    OUTPUT_TYPE out = TO_OUTPUT_TYPE(res);
#endif

#if OUTPUT_LAYOUT_BYXF_AF32 == 1
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, of, y, x);
#elif OUTPUT_LAYOUT_B_FS_YX_FSV4 == 1
    const uint dst_index = GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, b, of, y, x);
#else
    #error "Incorrect output layout"
#endif

    output[dst_index] = ACTIVATION(out, ACTIVATION_PARAMS);
}
