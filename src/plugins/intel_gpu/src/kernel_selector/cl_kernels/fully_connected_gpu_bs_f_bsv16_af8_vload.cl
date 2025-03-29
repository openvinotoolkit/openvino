// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/sub_group.cl"

// Block read - currently block is 4 bytes aligned.
#define ALIGNED_BLOCK_READ8(ptr, byte_offset) as_half8(_sub_group_block_read_us8((const __global ushort*)(ptr) + (byte_offset)))

#define MULTIPLY_BLOCKS_16x8(_result, _blockA, _blockB)  \
{   \
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

#define SUB_GROUP_SIZE 16
#define INPUT_FEATURE_ALIGNMENT 8
#define ALIGNED_INPUT0_ELEMENTS_COUNT ALIGN(INPUT0_ELEMENTS_COUNT, INPUT_FEATURE_ALIGNMENT)

__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, 1, 1)))
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL (fully_connected_gpu_bs_f_bsv16_af8_vload)(
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
    const uint group_id = get_group_id(0);
    const uint batch_group_id = get_global_id(1); // which part of batches we are computing,
                                                  // for example for batch 64 we compute batches 0..31 for batch_group_id == 0
                                                  // and batches 32..65 for batch_group_id == 1
    const uint id_in_sub_group = get_sub_group_local_id();

    const uint out_id = (id_in_sub_group * BATCHES_PER_WORK_ITEM * (uint)get_global_size(1)) / SUB_GROUP_SIZE +
                        group_id * BATCHES_PER_WORK_ITEM * (uint)get_global_size(1) +
                        (BATCHES_PER_WORK_ITEM * batch_group_id) / SUB_GROUP_SIZE;

    uint neuronIdx = id_in_sub_group + group_id * SUB_GROUP_SIZE;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 16) blockC00 = UNIT_VAL_ZERO;

    uint weight_offset = id_in_sub_group + SUB_GROUP_SIZE * group_id * ALIGNED_INPUT0_ELEMENTS_COUNT;

    uint input_idx = id_in_sub_group + batch_group_id * BATCHES_PER_WORK_ITEM * ALIGNED_INPUT0_ELEMENTS_COUNT;
    for(uint h = 0; h < INPUT0_ELEMENTS_COUNT / 8; h++)
    {
        // read input data in blocks ( 16 batch * 8 x )
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA00 = ALIGNED_BLOCK_READ8(input, input_idx);
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockB00 = ALIGNED_BLOCK_READ8(weight, weight_offset);

        MULTIPLY_BLOCKS_16x8(blockC00, blockA00, blockB00)

        weight_offset += 128;
        input_idx     += 128; // 128 = 16 x 8 - because of input format which have blocks of 128 elements
    }

#if ALIGNED_INPUT0_ELEMENTS_COUNT != INPUT0_ELEMENTS_COUNT
    {
        // Processing of leftover input features
        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockA00 = UNIT_VAL_ZERO;
        for (uint idx = 0; idx < INPUT0_ELEMENTS_COUNT % INPUT_FEATURE_ALIGNMENT; idx++) {
            blockA00[idx] = input[input_idx + idx * SUB_GROUP_SIZE];
        }

        MAKE_VECTOR_TYPE(UNIT_TYPE, 8) blockB00 = ALIGNED_BLOCK_READ8(weight, weight_offset);
        MULTIPLY_BLOCKS_16x8(blockC00, blockA00, blockB00)
    }
#endif

#if BIAS_TERM
    blockC00 += bias[neuronIdx];
#endif // #if BIAS_TERM

    blockC00 = ACTIVATION(blockC00, ACTIVATION_PARAMS);

    vstore16(blockC00, out_id, output);
}

#undef SUB_GROUP_SIZE
#undef ALIGNED_BLOCK_READ8
#undef MULTIPLY_BLOCKS_16x8
