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

#define FILTER_IFM_MMAD_NUM ((FILTER_IFM_NUM + 31) / 32)
#define FILTER_OFM_MMAD_NUM ((FILTER_OFM_NUM + 7) / 8)
#define FILTER_IFM_ALIGNED (FILTER_IFM_MMAD_NUM * 32)
#define FILTER_OFM_ALIGNED (FILTER_OFM_MMAD_NUM * 8)

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(convolution_MMAD_blocks)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights,
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    __global float* quantizations,
#if CALIBRATION_TERM
    __global float* calibrations,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0) * OUTPUT_BLOCK_WIDTH;
    const uint y = get_global_id(1) * OUTPUT_BLOCK_HEIGHT;
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = get_global_id(2) % FILTER_OFM_ALIGNED;
    const uint b = get_global_id(2) / FILTER_OFM_ALIGNED;
#endif
    
    // our output values
    int out[OUTPUT_BLOCK_WIDTH * OUTPUT_BLOCK_HEIGHT] = { 0 };
    int in[IN_BLOCK_ARRAY_SIZE];

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;

    const uint filter_offset = (get_group_id(2) % FILTER_OFM_MMAD_NUM) * FILTER_OFM_BLOCK_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    uint in_addr = input_offset + input_x * INPUT0_X_PITCH + input_y * INPUT0_Y_PITCH;
    uint filter_idx = filter_offset;

   	__attribute__((opencl_unroll_hint(1)))
    for (uint k = 0; k < FILTER_IFM_MMAD_NUM; ++k)
    {
        // preload input data
        for(uint in_block_pos = 0; in_block_pos < IN_BLOCK_ARRAY_SIZE; in_block_pos++)
        {
            uint block_x = in_block_pos % IN_BLOCK_WIDTH;
            uint block_y = in_block_pos / IN_BLOCK_WIDTH;
            uint input_idx = in_addr + block_x * INPUT0_X_PITCH + block_y * INPUT0_Y_PITCH;
            in[in_block_pos] = as_int(intel_sub_group_block_read((const __global uint*)(input + input_idx)));
        }    
        // end of preloading input data

        __attribute__((opencl_unroll_hint(FILTER_SIZE_Y)))
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
		    __attribute__((opencl_unroll_hint(FILTER_SIZE_X)))
            for (uint i = 0; i < FILTER_SIZE_X ; ++i)
            {
                int8 weights_data = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + filter_idx)));

			    __attribute__((opencl_unroll_hint(OUTPUT_BLOCK_HEIGHT)))
                for(uint br = 0; br < OUTPUT_BLOCK_HEIGHT; br++)
                {
				    __attribute__((opencl_unroll_hint(OUTPUT_BLOCK_WIDTH)))
                    for(uint bc = 0; bc < OUTPUT_BLOCK_WIDTH; bc++)
                    {
                        int input_data = in[(br * STRIDE_SIZE_Y + j) * IN_BLOCK_WIDTH + bc * STRIDE_SIZE_X + i];
                        int8 activations;  //activations of all lanes
                        activations.s0 = sub_group_broadcast(input_data, 0); 
                        activations.s1 = sub_group_broadcast(input_data, 1); 
                        activations.s2 = sub_group_broadcast(input_data, 2); 
                        activations.s3 = sub_group_broadcast(input_data, 3); 
                        activations.s4 = sub_group_broadcast(input_data, 4); 
                        activations.s5 = sub_group_broadcast(input_data, 5); 
                        activations.s6 = sub_group_broadcast(input_data, 6); 
                        activations.s7 = sub_group_broadcast(input_data, 7); 
        
                        out[br * OUTPUT_BLOCK_WIDTH + bc] = MMAD_8(activations, weights_data, out[br * OUTPUT_BLOCK_WIDTH + bc]);
                    }
                }
                filter_idx += 32*8; // 32 features per channel * 8 output features per SIMD channel
            }
        }
        in_addr += 32; // 4 features per channel * 8 SIMD channels
    }

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif

    for(uint br = 0; br < OUTPUT_BLOCK_HEIGHT; br++)
    {
        for(uint bc = 0; bc < OUTPUT_BLOCK_WIDTH; bc++)
        {
#if CALIBRATION_TERM
            out[br * OUTPUT_BLOCK_WIDTH + bc] = (UNIT_TYPE)round(((float)out[br * OUTPUT_BLOCK_WIDTH + bc] * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
            out[br * OUTPUT_BLOCK_WIDTH + bc] = (UNIT_TYPE)round(((float)out[br * OUTPUT_BLOCK_WIDTH + bc] * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM
        }
    }
#endif // BIAS_TERM

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    for(uint br = 0; br < OUTPUT_BLOCK_HEIGHT; br++)
    {
        if(y + br < OUTPUT_SIZE_Y)
        {
            for(uint bc = 0; bc < OUTPUT_BLOCK_WIDTH; bc++)
            {
                if(x + bc < OUTPUT_SIZE_X)
                {
                    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, y+br, x+bc) + out_split_offset;
                    output[dst_index] = ACTIVATION(convert_char(out[br * OUTPUT_BLOCK_WIDTH + bc]), NL_M, NL_N);
                }
            }
        }
    }
}

#undef FILTER_IFM_MMAD_NUM
#undef FILTER_OFM_MMAD_NUM
#undef FILTER_IFM_ALIGNED
#undef FILTER_OFM_ALIGNED