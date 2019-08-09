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
KERNEL(convolution_1x1_gemm_MMAD)(
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
    const uint sg_channel = get_sub_group_local_id();

    const uint x = (get_group_id(0) * 8) % INPUT0_SIZE_X;
    const uint y = (get_group_id(0) * 8) / INPUT0_SIZE_X;
    const uint f = get_global_id(1) % FILTER_OFM_ALIGNED;
    const uint b = get_global_id(1) / FILTER_OFM_ALIGNED;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;

    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;
    uint in_addr = input_offset + input_x * INPUT0_X_PITCH + input_y * INPUT0_Y_PITCH;

    const uint filter_offset = (get_group_id(1) % FILTER_OFM_MMAD_NUM) * FILTER_OFM_BLOCK_PITCH;
    uint filter_idx = filter_offset;

    int8 tileA;
    int8 tileB;
    int8 tileC;
    for(uint i = 0; i < 8; i++)
    {
        tileC[i] = 0;
    }

    for (uint k = 0; k < FILTER_IFM_MMAD_NUM; ++k)
    {
        // load A tile ( input )
        for(uint i = 0; i < 8; i++)
        {
            uint tmp_addr = in_addr + i * INPUT0_X_PITCH;
            tileA[i] = as_int(intel_sub_group_block_read((const __global uint*)(input + tmp_addr)));
        }

        // load B tile ( weights )
        tileB = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + filter_idx)));
    
        // compute C tile ( output )
        tileC = MMAD_8x8(tileA, tileB, tileC);

        in_addr += 32; // 4 features per channel * 8 SIMD channels
        filter_idx += 32*8; // 32 features per channel * 8 output features per SIMD channel
    }

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
#elif BIAS_PER_OFM
    const uint bias_index = f;
#endif
    for(uint i = 0; i < 8; i++)
    {
#if CALIBRATION_TERM
    tileC[i] = (UNIT_TYPE)round(((float)tileC[i] * quantizations[f] * I_QF + biases[bias_index]) * calibrations[f]);
#else  // CALIBRATION_TERM
    tileC[i] = (UNIT_TYPE)round(((float)tileC[i] * quantizations[f] * I_QF + biases[bias_index]) * O_QF);
#endif // CALIBRATION_TERM
    }
#endif // BIAS_TERM

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;
    // save to output
    for(uint i = 0; i < 8; i++)
    {
        const uint curr_x = (x + i) % INPUT0_SIZE_X;
        const uint curr_y = y + (x + i) / INPUT0_SIZE_X;
        if(curr_x < INPUT0_SIZE_X && curr_y < INPUT0_SIZE_Y)
        {
            const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, curr_y, curr_x) + out_split_offset;
            output[dst_index] = ACTIVATION(convert_char(tileC[i]), ACTIVATION_PARAMS);
        }
    }
}

#undef FILTER_IFM_MMAD_NUM
#undef FILTER_OFM_MMAD_NUM
#undef FILTER_IFM_ALIGNED
#undef FILTER_OFM_ALIGNED