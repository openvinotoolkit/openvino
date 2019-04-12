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
#include "include/sub_group.cl"

#if FP16_UNIT_USED
    #define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_half8(intel_sub_group_block_read_us8((const __global ushort*)(ptr) + (byte_offset)))
    
    #define MULTIPLY_BLOCKS_16x8_8x16(_result, _blockA, _blockB) \
    { \
        const half16 acol0 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s0 ); \
        const half16 acol1 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s1 ); \
        const half16 acol2 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s2 ); \
        const half16 acol3 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s3 ); \
        const half16 acol4 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s4 ); \
        const half16 acol5 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s5 ); \
        const half16 acol6 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s6 ); \
        const half16 acol7 = TRANSPOSE_BLOCK_16_FP16_HALF_TYPE( _blockA.s7 ); \
        _result = fma( _blockB.s0, acol0, _result ); \
        _result = fma( _blockB.s1, acol1, _result ); \
        _result = fma( _blockB.s2, acol2, _result ); \
        _result = fma( _blockB.s3, acol3, _result ); \
        _result = fma( _blockB.s4, acol4, _result ); \
        _result = fma( _blockB.s5, acol5, _result ); \
        _result = fma( _blockB.s6, acol6, _result ); \
        _result = fma( _blockB.s7, acol7, _result ); \
    }
#else
    // Block read - currently block is 4 bytes aligned.
    #define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_float8(intel_sub_group_block_read8((const __global uint*)(ptr) + (byte_offset)))

    #define MULTIPLY_BLOCKS_16x8_8x16(_result, _blockA, _blockB) \
    { \
        const float16 acol0 = TRANSPOSE_BLOCK_16( _blockA.s0 ); \
        const float16 acol1 = TRANSPOSE_BLOCK_16( _blockA.s1 ); \
        const float16 acol2 = TRANSPOSE_BLOCK_16( _blockA.s2 ); \
        const float16 acol3 = TRANSPOSE_BLOCK_16( _blockA.s3 ); \
        const float16 acol4 = TRANSPOSE_BLOCK_16( _blockA.s4 ); \
        const float16 acol5 = TRANSPOSE_BLOCK_16( _blockA.s5 ); \
        const float16 acol6 = TRANSPOSE_BLOCK_16( _blockA.s6 ); \
        const float16 acol7 = TRANSPOSE_BLOCK_16( _blockA.s7 ); \
        _result = fma( _blockB.s0, acol0, _result ); \
        _result = fma( _blockB.s1, acol1, _result ); \
        _result = fma( _blockB.s2, acol2, _result ); \
        _result = fma( _blockB.s3, acol3, _result ); \
        _result = fma( _blockB.s4, acol4, _result ); \
        _result = fma( _blockB.s5, acol5, _result ); \
        _result = fma( _blockB.s6, acol6, _result ); \
        _result = fma( _blockB.s7, acol7, _result ); \
    }
#endif

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL(convolution_bfyx_1x1)(
    __global INPUT0_TYPE* input, 
    __global OUTPUT_TYPE* output, 
    __global FILTER_TYPE* weights, 
#if BIAS_TERM
    __global BIAS_TYPE* biases,
#endif
    uint split_idx)
{
    const uint xy = get_group_id(0) * 16 + get_sub_group_local_id();
    const uint x = xy % OUTPUT_SIZE_X;
    const uint y = xy / OUTPUT_SIZE_X;
    const uint f = get_group_id(1) * 16 + get_sub_group_local_id();//get_global_id(1);
    const uint b = get_global_id(2);
    const uint group_f = get_group_id(1) * 16;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 16) blockC00 = UNIT_VAL_ZERO;

#if BIAS_TERM
    #if   BIAS_PER_OUTPUT
        const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
    #elif BIAS_PER_OFM
        const uint bias_index = f;
    #endif
    for(uint i = 0; i < 16; i++)
    {
        blockC00[i] = intel_sub_group_shuffle(biases[bias_index], i);
    }
#endif

#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    const uint filter_offset = group_f * ((FILTER_OFM_PITCH + 8 - 1) / 8) * 8;//f*FILTER_OFM_PITCH;
    const uint xy_block_num = (INPUT0_FEATURE_PITCH + 16 - 1) / 16;
    const uint f_block_num = (INPUT0_FEATURE_NUM + 8 - 1) / 8;
    const uint input_offset = in_split_offset + xy * 8 + b * xy_block_num * f_block_num * 128;//b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    for (uint k = 0; k < (FILTER_IFM_NUM + 8 - 1) / 8; ++k)
    {
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA00;
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockB00;
    
        uint input_idx = input_offset + k * 8 * xy_block_num * 16;
        uint filter_idx = filter_offset + k * 8 * 16;
    
        blockA00 = ALIGNED_BLOCK_READ8(input, input_idx);
        blockB00 = ALIGNED_BLOCK_READ8(weights, filter_idx);

        MULTIPLY_BLOCKS_16x8_8x16(blockC00, blockB00, blockA00);
    }

    if(xy >= INPUT0_SIZE_X * INPUT0_SIZE_Y)
        return;

    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;

    for(uint i = 0; i < 16; i++)
    {
    #if OUTPUT_LAYOUT_BF8_XY16
        const uint dst_index = GET_DATA_BF8_XY16_INDEX(OUTPUT, b, group_f+i, y, x) + out_split_offset;
    #else
        const uint dst_index = GET_DATA_INDEX(OUTPUT, b, group_f+i, y, x) + out_split_offset;
    #endif
    #if LEFTOVERS
        if(group_f+i < OUTPUT_FEATURE_NUM)
    #endif
        output[dst_index] = ACTIVATION(blockC00[i], NL_M, NL_N);   
    }
}

#undef ALIGNED_BLOCK_READ8
#undef MULTIPLY_BLOCKS_16x8_8x16
#undef CONCAT_TOKEN
#undef CONCAT_TOKEN_HANDLER1
#undef MULTIPLY_BLOCKS_16x16
#undef MAKE_VECTOR_TYPE