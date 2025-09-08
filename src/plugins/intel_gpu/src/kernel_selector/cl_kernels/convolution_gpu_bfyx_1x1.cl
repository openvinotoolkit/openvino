// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/sub_group.cl"

#if FP16_UNIT_USED
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

#ifndef ACCUMULATOR_TYPE
#define ACCUMULATOR_TYPE INPUT0_TYPE
#endif

REQD_SUB_GROUP_SIZE(16)
KERNEL(convolution_bfyx_1x1)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
)
{
    const uint group_xy = (uint)get_group_id(0) * 16;
    const uint xy = group_xy + get_sub_group_local_id();
    const uint x = xy % OUTPUT_SIZE_X;
    const uint y = xy / OUTPUT_SIZE_X;
    const uint f = (uint)get_group_id(1) * 16 + get_sub_group_local_id();//get_global_id(1);
    const uint b = (uint)get_global_id(2);
    const uint group_f = (uint)get_group_id(1) * 16;

    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 16) blockC00 = INPUT0_VAL_ZERO;

#if BIAS_TERM
    #if   BIAS_PER_OUTPUT
        const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);
    #elif BIAS_PER_OFM
        const uint bias_index = f;
    #endif
    for(uint i = 0; i < 16; i++)
    {
        blockC00[i] = _sub_group_shuffle(biases[bias_index], i);
    }
#endif

    const uint filter_offset = group_f * ((FILTER_OFM_PITCH + 8 - 1) / 8) * 8;//f*FILTER_OFM_PITCH;
    const uint xy_block_num = (INPUT0_FEATURE_PITCH + 16 - 1) / 16;
    const uint f_block_num = (INPUT0_FEATURE_NUM + 8 - 1) / 8;
    const uint input_offset = group_xy * 8 + b * xy_block_num * f_block_num * 128;//b*INPUT0_BATCH_PITCH + INPUT0_OFFSET;

    for (uint k = 0; k < (FILTER_IFM_NUM + 8 - 1) / 8; ++k)
    {
        MAKE_VECTOR_TYPE(INPUT0_TYPE, 8) blockA00;
        MAKE_VECTOR_TYPE(FILTER_TYPE, 8) blockB00;

        uint input_idx = input_offset + k * 8 * xy_block_num * 16;
        uint filter_idx = filter_offset + k * 8 * 16;

        blockA00 = DT_INPUT_BLOCK_READ8(input, input_idx);
        blockB00 = DT_FILTER_BLOCK_READ8(weights, filter_idx);

        MULTIPLY_BLOCKS_16x8_8x16(blockC00, blockB00, blockA00);
    }

    if(xy >= INPUT0_SIZE_X * INPUT0_SIZE_Y)
        return;

    for(uint i = 0; i < 16; i++)
    {
        const uint dst_index = GET_DATA_INDEX(OUTPUT, b, group_f+i, y, x);
    #if LEFTOVERS
        if(group_f+i < OUTPUT_FEATURE_NUM)
    #endif
        output[dst_index] = ACTIVATION(blockC00[i], ACTIVATION_PARAMS);
    }
}

#undef ALIGNED_BLOCK_READ8
#undef MULTIPLY_BLOCKS_16x8_8x16
#undef CONCAT_TOKEN
#undef CONCAT_TOKEN_HANDLER1
#undef MULTIPLY_BLOCKS_16x16
#undef ACCUMULATOR_TYPE
