// Copyright (c) 2018 Intel Corporation
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

#if FP16_UNIT_USED
    #define ALIGNED_BLOCK_READ(ptr, byte_offset) as_half(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (byte_offset)))
    #define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write_us((__global ushort*)(ptr) + (byte_offset), as_ushort8(val))
#else
    #define ALIGNED_BLOCK_READ(ptr, byte_offset) as_float(intel_sub_group_block_read((const __global uint*)(ptr) + (byte_offset)))
    #define ALIGNED_BLOCK_WRITE(ptr, byte_offset, val) intel_sub_group_block_write((__global uint*)(ptr) + (byte_offset), as_uint8(val))
#endif

__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
KERNEL(convolution_depthwise_weights_lwg)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx)
{
    const uint yx = get_global_id(0);
    const uint x = yx % OUTPUT_SIZE_X;
    const uint y = yx / OUTPUT_SIZE_X;
    const uint f = get_global_id(1);
    const uint b = get_global_id(2);

    UNIT_TYPE dotProd = UNIT_VAL_ZERO;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;

    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

#if FILTER_SIZE_Y * FILTER_SIZE_X % 16 == 0 && !FP16_UNIT_USED
    UNIT_TYPE w = ALIGNED_BLOCK_READ(weights, filter_offset);
#else
    const uint lid = get_local_id(0);
    UNIT_TYPE w = UNIT_VAL_ZERO;
    if(lid < FILTER_SIZE_X * FILTER_SIZE_Y)
        w = weights[filter_offset + lid];
#endif

    __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
    for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
    {
        const int input_offset_y = input_y + j * DILATION_SIZE_Y;
#if BOUNDARY_CHECK
        const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

        if(!zero_y)
        {
#endif
            __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (uint i = 0; i < FILTER_SIZE_X ; ++i)
            {
                const int input_offset_x = input_x + i * DILATION_SIZE_X;
#if BOUNDARY_CHECK
                const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                if(!zero_x)
                {
#endif
                    dotProd = mad(input[input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH],
                                  intel_sub_group_shuffle( w, j*FILTER_Y_PITCH + i*FILTER_X_PITCH), dotProd);
                }
            }
#if BOUNDARY_CHECK
        }
    }
#endif

    if(yx >= OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
        return;

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif
    dotProd += (UNIT_TYPE)biases[bias_index];
#endif

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, y, x) + out_split_offset;
    output[dst_index] = ACTIVATION(dotProd, NL_M, NL_N);
    
}
