// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/reshape_dims.cl"
#include "include/batch_headers/data_types.cl"

#define OFM_BLOCK_SIZE 32
#define IFM_PACK_SIZE 32

// packed binary oihw to packed binary os_is_yx_osv32_isv32p
KERNEL (reorder_weights_binary)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const unsigned o = get_global_id(0);
    const unsigned i = get_global_id(1);
    const unsigned y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const unsigned x = (uint)get_global_id(2) % OUTPUT_SIZE_X;

    int output_index = OUTPUT_OFFSET
                     + (o % OFM_BLOCK_SIZE)
                     + (o / OFM_BLOCK_SIZE) * ((OUTPUT_IFM_NUM + IFM_PACK_SIZE - 1) / IFM_PACK_SIZE) * OUTPUT_SIZE_Y * OUTPUT_SIZE_X * OFM_BLOCK_SIZE
                     + i * OFM_BLOCK_SIZE * OUTPUT_IFM_PITCH
                     + y * OFM_BLOCK_SIZE * OUTPUT_Y_PITCH
                     + x * OFM_BLOCK_SIZE * OUTPUT_X_PITCH;

    OUTPUT_TYPE res = 0x00000000;
    int limit = min((int)IFM_PACK_SIZE, (int)(INPUT0_IFM_NUM - i*IFM_PACK_SIZE));
    for (int c = 0; c < limit; c++)
    {
        // index of required bit
        int input_index = INPUT0_OFFSET
                        + o * INPUT0_OFM_PITCH
                        + (i * IFM_PACK_SIZE + c) * INPUT0_IFM_PITCH
                        + y * INPUT0_Y_PITCH
                        + x * INPUT0_X_PITCH;

        const int bit = input_index % IFM_PACK_SIZE;
        const int element = input_index / IFM_PACK_SIZE;
        res |= ((input[element] & (1 << bit)) >> bit) << c;
    }

    output[output_index] = res;
}
