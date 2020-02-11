// Copyright (c) 2019 Intel Corporation
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
#include "include/unit_type.cl"
#include "include/sub_group.cl"

#if FP16_UNIT_USED
    #define MAD_1X8(_result_block, _input_value, _weights_block) \
    { \
        _result_block.s0 = fma(_input_value, _weights_block.s0, _result_block.s0); \
        _result_block.s1 = fma(_input_value, _weights_block.s1, _result_block.s1); \
        _result_block.s2 = fma(_input_value, _weights_block.s2, _result_block.s2); \
        _result_block.s3 = fma(_input_value, _weights_block.s3, _result_block.s3); \
        _result_block.s4 = fma(_input_value, _weights_block.s4, _result_block.s4); \
        _result_block.s5 = fma(_input_value, _weights_block.s5, _result_block.s5); \
        _result_block.s6 = fma(_input_value, _weights_block.s6, _result_block.s6); \
        _result_block.s7 = fma(_input_value, _weights_block.s7, _result_block.s7); \
    }
#else
    #define MAD_1X8(_result_block, _input_value, _weights_block) \
    { \
        _result_block.s0 = mad(_input_value, _weights_block.s0, _result_block.s0); \
        _result_block.s1 = mad(_input_value, _weights_block.s1, _result_block.s1); \
        _result_block.s2 = mad(_input_value, _weights_block.s2, _result_block.s2); \
        _result_block.s3 = mad(_input_value, _weights_block.s3, _result_block.s3); \
        _result_block.s4 = mad(_input_value, _weights_block.s4, _result_block.s4); \
        _result_block.s5 = mad(_input_value, _weights_block.s5, _result_block.s5); \
        _result_block.s6 = mad(_input_value, _weights_block.s6, _result_block.s6); \
        _result_block.s7 = mad(_input_value, _weights_block.s7, _result_block.s7); \
    }
#endif

#define INC_OFFSET(_offset, _value) _offset += _value
#define SIMD_SIZE 8

__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL(lstm_dynamic_input_bfyx_opt)(
    const __global INPUT0_TYPE* input,
    const __global DYN_LENGTH_TYPE* dyn_lengths,
    __global OUTPUT_TYPE* output,
    const __global WEIGHTS_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
    )
{
    const uint batch    = (uint)get_global_id(1) % INPUT0_BATCH_NUM;
    const uint dir      = (uint)get_global_id(1) / INPUT0_BATCH_NUM;
    const uint timestep = get_global_id(2);
    if(timestep > (uint)dyn_lengths[batch])
        return;
    // which general local work item within work group we have
    const uint local_work_item_id = get_local_id(0);
    // which id in SUBGROUP we have (0..7)
    const uint sub_group_local_id = get_sub_group_local_id();
    // which SUBGROUP we have
    const uint sub_group_id     = local_work_item_id / SIMD_SIZE;//get_sub_group_id();
    const uint dir_sub_group_id = sub_group_id % SIMD_SIZE;
    //which workgroup we have <0,1>
    const uint wg_id     = get_group_id(0);
    const uint wg_offset = wg_id * (uint)get_local_size(0) * SIMD_SIZE;
    //Subgroups have region of calcuations (ROC) within each local work item calculate simd_size values across y spatial.
    //i.e sub_group_id = 1 have ROC, which starts at 64th y'th position
    const uint sub_group_offset        = SIMD_SIZE * 8;
    const uint weights_single_dir_size = WEIGHTS_SIZE_X * WEIGHTS_SIZE_Y;
    const uint dir_offset_for_weights  = dir * weights_single_dir_size;
    uint calcuation_offset      = dir_offset_for_weights + wg_offset + dir_sub_group_id * sub_group_offset;
    uint input_offset           = GET_DATA_INDEX(INPUT0, batch, timestep, dir, sub_group_local_id);
    const uint output_offset    = GET_DATA_INDEX(OUTPUT, batch, timestep, dir, wg_offset + dir_sub_group_id * sub_group_offset);

#if BIAS_TERM
    //preload output with biases
    const uint bias_calcuation_offset  = dir * BIAS_SIZE_X + wg_offset + dir_sub_group_id * sub_group_offset;
    UNIT_TYPE8 dot_prod = UNIT_BLOCK_READ8(biases, bias_calcuation_offset);
#else
    UNIT_TYPE8 dot_prod = UNIT_VAL_ZERO;
#endif

    for(uint x = 0; x < INPUT0_SIZE_X / SIMD_SIZE; ++x)
    {
        UNIT_TYPE8 BLOCK_W0 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
        UNIT_TYPE8 BLOCK_W1 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
        UNIT_TYPE8 BLOCK_W2 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
        UNIT_TYPE8 BLOCK_W3 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
        UNIT_TYPE8 BLOCK_W4 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
        UNIT_TYPE8 BLOCK_W5 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
        UNIT_TYPE8 BLOCK_W6 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
        UNIT_TYPE8 BLOCK_W7 = UNIT_BLOCK_READ8(weights, calcuation_offset); INC_OFFSET(calcuation_offset, WEIGHTS_SIZE_Y);
            
        UNIT_TYPE input_value = input[input_offset];
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 0), BLOCK_W0);
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 1), BLOCK_W1);
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 2), BLOCK_W2);
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 3), BLOCK_W3);
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 4), BLOCK_W4);
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 5), BLOCK_W5);
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 6), BLOCK_W6);
        MAD_1X8(dot_prod, intel_sub_group_shuffle(input_value, 7), BLOCK_W7);
        
        input_offset += SIMD_SIZE;
    }

    UNIT_BLOCK_WRITE8(output, output_offset, dot_prod);
}

#undef SIMD_SIZE
#undef INC_OFFSET
#undef MAD_1X8
#undef OPT
