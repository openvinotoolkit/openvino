// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#define VLOAD CAT(vload, VEC_BLK_SIZE)
#define VSTORE CAT(vstore, VEC_BLK_SIZE)

// vload/vstore only assume the alignment of one element. When rows are a multiple of VEC_BLK_SIZE elements, every
// block starts at a multiple of its own size, and a vector pointer lets the compiler use full-width block accesses.
#if HIDDEN_SIZE % VEC_BLK_SIZE == 0
#    define VEC_TYPE CAT(INPUT0_TYPE, VEC_BLK_SIZE)
#    define COPY_BLOCK(src, dst) (*(__global VEC_TYPE*)(dst) = *(const __global VEC_TYPE*)(src))
#else
#    define COPY_BLOCK(src, dst) VSTORE(VLOAD(0, src), 0, dst)
#endif

KERNEL(moe_gather_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* token_indices,
    __global OUTPUT_TYPE* output
)
{
    const uint token_group_id = (uint)get_group_id(0);
    const uint threads_index = (uint)get_local_id(0);

#if UNALIGNED_ELEMENTS > 0
    if ((threads_index == get_local_size(0) - 1) && (UNALIGNED_ELEMENTS > 0)) {
        for (uint i = 0; i < UNALIGNED_ELEMENTS; i++) {
            const INPUT1_TYPE token_index = token_indices[token_group_id] * HIDDEN_SIZE;

            const uint dest_index = token_group_id * HIDDEN_SIZE;

            const uint input_pos = token_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD + i;
            const uint output_pos = dest_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD + i;
            output[output_pos] = input[input_pos];
        }
    } else {
#endif
        for (uint i = 0; i < BATCHES_PER_THREAD; i++) {
            const INPUT1_TYPE token_index = token_indices[token_group_id] * HIDDEN_SIZE  +  i * VEC_BLK_SIZE;

            const uint dest_index = token_group_id * HIDDEN_SIZE + i * VEC_BLK_SIZE;

            const uint input_pos = token_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;
            const uint output_pos = dest_index + threads_index * VEC_BLK_SIZE * BATCHES_PER_THREAD;
            COPY_BLOCK(&input[input_pos], &output[output_pos]);
        }
#if UNALIGNED_ELEMENTS > 0
    }
#endif
}
