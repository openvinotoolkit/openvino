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

#define OBS 8
__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(convolution)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
#if QUANTIZATION_TERM
    __global float* quantizations,
#endif
#if CALIBRATION_TERM
    __global float* calibrations,
#endif
    uint split_idx)
{
    const uint f_pack = (get_group_id(0) * 32) % OUTPUT_FEATURE_NUM;
    const uint b = (get_group_id(0) * 32) / OUTPUT_FEATURE_NUM;

    const uint x = get_group_id(1) * OBS;
    const uint y = get_group_id(2);

    int4 dotProd[OBS] = { 0 };

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = f_pack*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET;

    for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
    {
        const int input_offset_y = input_y + j;
        for (uint i = 0; i < FILTER_SIZE_X ; ++i)
        {
            const int input_offset_x = input_x + i + STRIDE_SIZE_X * get_sub_group_local_id();
            uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH;
            uint filter_idx = filter_offset + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;

            char input_data[3];
            char2 _i = vload2(0, input + input_idx);
            input_data[0] = _i.s0;
            input_data[1] = _i.s1;
            input_data[2] = input[input_idx + 2];

            for (uint k = 0; k < FILTER_IFM_NUM; ++k)
            {
                char4 w_data = as_char4(intel_sub_group_block_read((const __global uint*)(weights + filter_idx)));
                for(uint r = 0; r < OBS; r++)
                {
                    char in = intel_sub_group_shuffle(input_data[k], r);
                    for(uint c = 0; c < 4; c++)
                    {
                        dotProd[r][c] += (int)in * (int)w_data[c];
                    }
                }
                filter_idx += FILTER_IFM_PITCH;
            }
        }
    }


const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, b, f_pack, y, x + get_sub_group_local_id());
const uint _f_idx = f_pack + get_sub_group_local_id() * 4;
float4 quants = vload4(0, quantizations + _f_idx );
float4 calibs = vload4(0, calibrations + _f_idx );
float4 bias = vload4(0, biases + _f_idx );
for(uint r = 0; r < OBS; r++)
{
    char4 char_output;
    for(uint c = 0; c < 4; c++)
    {
        const uint f_idx = f_pack + get_sub_group_local_id() * 4 + c;
    #if BIAS_TERM
        const uint bias_index = f_idx;
    #if CALIBRATION_TERM
        dotProd[r][c] = (UNIT_TYPE)round(((float)dotProd[r][c] * quants[c] * I_QF + bias[c]) * calibs[c]);
    #else  // CALIBRATION_TERM
        dotProd[r][c] = (UNIT_TYPE)round(((float)dotProd[r][c] * quants[c] * I_QF + bias[c]) * O_QF);
    #endif // CALIBRATION_TERM
    #endif
        char_output[c] = ACTIVATION(convert_char(dotProd[r][c]), ACTIVATION_PARAMS);
    }
    const uint out_idx = intel_sub_group_shuffle(dst_index, r);
    intel_sub_group_block_write( (__global uint*)(output + out_idx) , as_uint(char_output));
}

}
