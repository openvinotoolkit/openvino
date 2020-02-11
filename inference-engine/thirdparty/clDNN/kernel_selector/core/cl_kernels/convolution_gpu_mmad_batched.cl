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

#include "include/common.cl"

#include "include/data_types.cl"
#include "include/fetch.cl"
#include "include/mmad.cl"

#define FILTER_IFM_MMAD_NUM ((FILTER_IFM_NUM + 31) / 32)
#define FILTER_OFM_MMAD_NUM ((FILTER_OFM_NUM + 7) / 8)
#define FILTER_IFM_ALIGNED (FILTER_IFM_MMAD_NUM * 32)
#define FILTER_OFM_ALIGNED (FILTER_OFM_MMAD_NUM * 8)
// input data is in blocks 4batch x 32 features
// each SIMD process 4 batches and 8 output features

__attribute__((intel_reqd_sub_group_size(SUB_GROUP_SIZE)))
KERNEL(convolution_mmad_batched)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if QUANTIZATION_TERM
    const __global float* quantizations,
#endif
#if CALIBRATION_TERM
    const __global float* calibrations,
#endif
    uint split_idx)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);

    const uint f = (uint)get_global_id(2) % FILTER_OFM_ALIGNED;
    const uint b_block = (uint)get_global_id(2) / FILTER_OFM_ALIGNED;
    const uint f_block = f / 32;

    int4 dotProd = 0;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = ((uint)get_group_id(2) % FILTER_OFM_MMAD_NUM) * FILTER_OFM_BLOCK_PITCH;
    const uint input_offset = IN_OFFSET + IN_B_BLOCK_PITCH * b_block;

    for (uint k = 0; k < FILTER_IFM_MMAD_NUM; ++k)
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
                        uint input_idx = input_offset + input_offset_y * IN_Y_PITCH + input_offset_x * IN_X_PITCH + k * IN_F_BLOCK_PITCH;
                        uint filter_idx = filter_offset + k*FILTER_Y_PITCH * FILTER_SIZE_Y + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;

						int4 input_data = as_int4(intel_sub_group_block_read4((const __global uint*)(input + input_idx)));
                        int8 weights_data = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + filter_idx)));

                        dotProd = MMAD_4x8(input_data, weights_data, dotProd);
                    }
                }
            }
        }
    }

for(uint b = 0; b < 4; b++)
{

#if BIAS_TERM
    const uint bias_index = f;
#if QUANTIZATION_TERM
#if CALIBRATION_TERM
    dotProd[b] = (UNIT_TYPE)round(((float)dotProd[b] * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
    dotProd[b] = (UNIT_TYPE)round(((float)dotProd[b] * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM
#else // QUANTIZATION_TERM
    dotProd[b] += (UNIT_TYPE)biases[bias_index];
#endif // QUANTIZATION_TERM
#endif // BIAS_TERM

    const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b_block*4 + b, f, y, x);
#if QUANTIZATION_TERM
    output[dst_index] = ACTIVATION(convert_char(dotProd[b]), ACTIVATION_PARAMS);
#else
    output[dst_index] = ACTIVATION(dotProd[b], ACTIVATION_PARAMS);
#endif  
}
}

#undef FILTER_IFM_MMAD_NUM
#undef FILTER_OFM_MMAD_NUM
#undef FILTER_IFM_ALIGNED
#undef FILTER_OFM_ALIGNED
