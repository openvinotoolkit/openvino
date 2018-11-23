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

__attribute__((intel_reqd_sub_group_size(8)))
KERNEL(fully_connected_kernel_mmad_batched)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if QUANTIZATION_TERM
    ,const __global float* quantizations
#endif
#if CALIBRATION_TERM
    ,const __global float* calibrations
#endif
    )
{
    const uint sg_channel = get_sub_group_local_id();

    const uint batch_id = (uint)get_group_id(0) * 8;
    const uint b_block = batch_id / 4;
    const uint f = get_global_id(1) % FILTER_OFM_ALIGNED;

    uint in_addr = IN_OFFSET + b_block * IN_B_BLOCK_PITCH;

    const uint filter_offset = (get_group_id(1) % FILTER_OFM_MMAD_NUM) * FILTER_OFM_BLOCK_PITCH;
    uint filter_idx = filter_offset;

    int8 tileA;
    int8 tileB;
    int8 tileC = 0;

    for(uint z = 0; z < FILTER_IFM_MMAD_NUM; z++ )
    {
        for (uint k = 0; k < FILTER_SIZE_X * FILTER_SIZE_Y; ++k)
        {
            // load A tile ( input )
            // load 8 batches 4 channels per WI, so we'll have 8x32 block

            tileA.lo = as_int4(intel_sub_group_block_read4((const __global uint*)(input + in_addr)));
            tileA.hi = as_int4(intel_sub_group_block_read4((const __global uint*)(input + in_addr + IN_B_BLOCK_PITCH)));

            // load B tile ( weights )
            tileB = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + filter_idx)));
    
            // compute C tile ( output )
            tileC = MMAD_8x8(tileA, tileB, tileC); // here we output 8 batches per workitem, and each workitem gets different output feature

            in_addr += 32 * 4; // 4 batches * 4 features per channel * 8 SIMD channels
            filter_idx += 32*8; // 32 features per channel * 8 output features per SIMD channel
        }
        in_addr += IN_F_BLOCK_PITCH;
        in_addr -= (FILTER_SIZE_X * FILTER_SIZE_Y * 32 * 4);
    }

#if BIAS_TERM
#if   BIAS_PER_OUTPUT
    const uint bias_index = GET_DATA_INDEX(BIAS, batch_id, f, y, x);
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

    // save to output
    if(f < FILTER_OFM_NUM)
    {
        for(uint i = 0; i < 8; i++)
        {
            const uint curr_b = batch_id + i;
#if defined OUTPUT_LAYOUT_FS_BS_YX_BSV4_FSV32
            const uint dst_index = GET_DATA_FS_BS_YX_BSV4_FSV32_INDEX(OUTPUT, curr_b, f, 0, 0);
#else
            const uint dst_index = GET_DATA_INDEX(OUTPUT, curr_b, f, 0, 0);
#endif
            output[dst_index] = ACTIVATION(convert_char(tileC[i]), NL_M, NL_N);
        }
    }
}

#undef FILTER_IFM_MMAD_NUM
#undef FILTER_OFM_MMAD_NUM
#undef FILTER_IFM_ALIGNED
#undef FILTER_OFM_ALIGNED