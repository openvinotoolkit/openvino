// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/sub_group.cl"

#if FP16_UNIT_USED
    #define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB)  \
    {   \
        const half8 acol0 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 0 ); \
        const half8 acol1 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 1 ); \
        const half8 acol2 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 2 ); \
        const half8 acol3 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 3 ); \
        const half8 acol4 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 4 ); \
        const half8 acol5 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 5 ); \
        const half8 acol6 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 6 ); \
        const half8 acol7 = TRANSPOSE_BLOCK_8_COL_FP16( _blockA, 7 ); \
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
    #define MULTIPLY_BLOCKS_8x8(_result, _blockA, _blockB)  \
    {   \
        const float8 acol0 = TRANSPOSE_BLOCK_8_COL( _blockA, 0 ); \
        const float8 acol1 = TRANSPOSE_BLOCK_8_COL( _blockA, 1 ); \
        const float8 acol2 = TRANSPOSE_BLOCK_8_COL( _blockA, 2 ); \
        const float8 acol3 = TRANSPOSE_BLOCK_8_COL( _blockA, 3 ); \
        const float8 acol4 = TRANSPOSE_BLOCK_8_COL( _blockA, 4 ); \
        const float8 acol5 = TRANSPOSE_BLOCK_8_COL( _blockA, 5 ); \
        const float8 acol6 = TRANSPOSE_BLOCK_8_COL( _blockA, 6 ); \
        const float8 acol7 = TRANSPOSE_BLOCK_8_COL( _blockA, 7 ); \
        _result = mad( _blockB.s0, acol0, _result ); \
        _result = mad( _blockB.s1, acol1, _result ); \
        _result = mad( _blockB.s2, acol2, _result ); \
        _result = mad( _blockB.s3, acol3, _result ); \
        _result = mad( _blockB.s4, acol4, _result ); \
        _result = mad( _blockB.s5, acol5, _result ); \
        _result = mad( _blockB.s6, acol6, _result ); \
        _result = mad( _blockB.s7, acol7, _result ); \
    }
#endif

#define SUB_GROUP_SIZE 8

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
KERNEL (fully_connected_gpu_xb_xb_b8_x8_vload)(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* weight
#if BIAS_TERM
    , __global UNIT_TYPE* bias)
#else
    )
#endif
{
    const uint global_id = get_global_id(0);
    const uint group_id = get_global_id(1); // which part of batches we are computing,
                                            // for example for batch 64 we compute batches 0..31 for group_id == 0
                                            // and batches 32..65 for group_id == 1
    uint sub_group_idx = (uint)get_local_id(0) % 8;

    const uint out_id = (sub_group_idx * BATCHES_PER_WORK_ITEM * (uint)get_global_size(1)) / 8 +
                        (global_id / 8) * BATCHES_PER_WORK_ITEM * NEURONS_PER_WORK_ITEM * (uint)get_global_size(1) +
                        (BATCHES_PER_WORK_ITEM * group_id) / 8;

    uint neuronIdx = sub_group_idx + (global_id / 8) * 8 * NEURONS_PER_WORK_ITEM;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC00 = UNIT_VAL_ZERO;
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC10 = UNIT_VAL_ZERO;

#if BATCHES_PER_WORK_ITEM >= 16
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC01 = UNIT_VAL_ZERO;
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC11 = UNIT_VAL_ZERO;
#endif

#if BATCHES_PER_WORK_ITEM >= 32
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC02 = UNIT_VAL_ZERO;
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC12 = UNIT_VAL_ZERO;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC03 = UNIT_VAL_ZERO;
    MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockC13 = UNIT_VAL_ZERO;
#endif

    uint weight_offset = neuronIdx;
#if NEURONS_PER_WORK_ITEM > 1

    uint weight_offset2 = neuronIdx + 8;

#endif // #if NEURONS_PER_WORK_ITEM > 1

    uint input_idx = sub_group_idx * (BATCHES_PER_WORK_ITEM / 8) * (uint)get_global_size(1) + (group_id * BATCHES_PER_WORK_ITEM) / 8;
    for (uint h = 0; h < INPUT0_ELEMENTS_COUNT / 8; h++)
    {
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA00 = vload8(input_idx, input);

#if BATCHES_PER_WORK_ITEM >= 16
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA01 = vload8(input_idx + 1, input);
#endif

#if BATCHES_PER_WORK_ITEM >= 32
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA02 = vload8(input_idx + 2, input);
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA03 = vload8(input_idx + 3, input);
#endif
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockB00;
        blockB00.s0 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        blockB00.s1 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        blockB00.s2 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        blockB00.s3 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        blockB00.s4 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        blockB00.s5 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        blockB00.s6 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        blockB00.s7 = weight[weight_offset]; weight_offset += FILTER_OFM_NUM;
        MULTIPLY_BLOCKS_8x8(blockC00, blockA00, blockB00)

#if BATCHES_PER_WORK_ITEM >= 16
        MULTIPLY_BLOCKS_8x8(blockC01, blockA01, blockB00)
#endif

#if BATCHES_PER_WORK_ITEM >= 32
        MULTIPLY_BLOCKS_8x8(blockC02, blockA02, blockB00)
        MULTIPLY_BLOCKS_8x8(blockC03, blockA03, blockB00)
#endif

#if NEURONS_PER_WORK_ITEM > 1

        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockB10;
        blockB10.s0 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        blockB10.s1 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        blockB10.s2 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        blockB10.s3 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        blockB10.s4 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        blockB10.s5 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        blockB10.s6 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        blockB10.s7 = weight[weight_offset2]; weight_offset2 += FILTER_OFM_NUM;
        MULTIPLY_BLOCKS_8x8(blockC10, blockA00, blockB10)

#if BATCHES_PER_WORK_ITEM >= 16
        MULTIPLY_BLOCKS_8x8(blockC11, blockA01, blockB10)
#endif
#if BATCHES_PER_WORK_ITEM >= 32
        MULTIPLY_BLOCKS_8x8(blockC12, blockA02, blockB10)
        MULTIPLY_BLOCKS_8x8(blockC13, blockA03, blockB10)
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
        input_idx += INPUT0_BATCH_NUM; // we don't need to multiply by 8 because of vload8
    }

#if BIAS_TERM
    blockC00 += bias[neuronIdx];
#if BATCHES_PER_WORK_ITEM >= 16
    blockC01 += bias[neuronIdx];
#endif

#if BATCHES_PER_WORK_ITEM >= 32
    blockC02 += bias[neuronIdx];
    blockC03 += bias[neuronIdx];
#endif

#if NEURONS_PER_WORK_ITEM > 1

    blockC10 += bias[neuronIdx + 8];
#if BATCHES_PER_WORK_ITEM >= 16
    blockC11 += bias[neuronIdx + 8];
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    blockC12 += bias[neuronIdx + 8];
    blockC13 += bias[neuronIdx + 8];
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
#endif // #if BIAS_TERM

    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);
#if BATCHES_PER_WORK_ITEM >= 16
    blockC01 = ACTIVATION(blockC01, ACTIVATION_PARAMS);
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    blockC02 = ACTIVATION(blockC02, ACTIVATION_PARAMS);
    blockC03 = ACTIVATION(blockC03, ACTIVATION_PARAMS);
#endif

#if NEURONS_PER_WORK_ITEM > 1

    blockC10 = ACTIVATION(blockC10, ACTIVATION_PARAMS);
#if BATCHES_PER_WORK_ITEM >= 16
    blockC11 = ACTIVATION(blockC11, ACTIVATION_PARAMS);
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    blockC12 = ACTIVATION(blockC12, ACTIVATION_PARAMS);
    blockC13 = ACTIVATION(blockC13, ACTIVATION_PARAMS);
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1

    vstore8(blockC00, out_id, output);
#if BATCHES_PER_WORK_ITEM >= 16
    vstore8(blockC01, out_id + 1, output);
#endif
#if BATCHES_PER_WORK_ITEM >= 32
    vstore8(blockC02, out_id + 2, output);
    vstore8(blockC03, out_id + 3, output);
#endif

#if NEURONS_PER_WORK_ITEM > 1

    vstore8(blockC10, out_id + INPUT0_BATCH_NUM, output);

#if BATCHES_PER_WORK_ITEM >= 16
    vstore8(blockC11, out_id + INPUT0_BATCH_NUM + 1, output);
#endif

#if BATCHES_PER_WORK_ITEM >= 32
    vstore8(blockC12, out_id + INPUT0_BATCH_NUM + 2, output);
    vstore8(blockC13, out_id + INPUT0_BATCH_NUM + 3, output);
#endif

#endif // #if NEURONS_PER_WORK_ITEM > 1
}

#undef SUB_GROUP_SIZE
#undef ALIGNED_BLOCK_READ8
#undef CONCAT_TOKEN
#undef CONCAT_TOKEN_HANDLER1
#undef MULTIPLY_BLOCKS_8x8
