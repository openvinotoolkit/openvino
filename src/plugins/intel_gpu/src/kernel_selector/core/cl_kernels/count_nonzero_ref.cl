// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"

#define INPUT0_GET_INDEX1(idx_order) INPUT0_GET_INDEX(idx_order)

KERNEL (count_nonzero_ref)(const __global INPUT0_TYPE* input,
                           volatile __global OUTPUT_TYPE* output)
{
    const uint gdim0 = (uint)get_global_id(0);
    const uint gdim1 = (uint)get_global_id(1);
    const uint gdim2 = (uint)get_global_id(2);

    #if INPUT0_DIMS == 6
        #define INPUT_ORDER b,f,w,z,y,x
        const uint x = gdim0 % INPUT0_SIZE_X;
        const uint y = gdim0 / INPUT0_SIZE_X;
        const uint z = gdim1 % INPUT0_SIZE_Z;
        const uint w = gdim1 / INPUT0_SIZE_Z;
    #elif INPUT0_DIMS == 5
        #define INPUT_ORDER b,f,z,y,x
        const uint x = gdim0;
        const uint y = gdim1 % INPUT0_SIZE_Y;
        const uint z = gdim1 / INPUT0_SIZE_Y;
    #elif INPUT0_DIMS == 4
        #define INPUT_ORDER b,f,y,x
        const uint x = gdim0;
        const uint y = gdim1;
    #endif
    const uint f = gdim2 % INPUT0_FEATURE_NUM;
    const uint b = gdim2 / INPUT0_FEATURE_NUM;

    uint count = (input[INPUT0_GET_INDEX1(INPUT_ORDER)] == INPUT0_VAL_ZERO) ? 0 : 1;
    count = sub_group_reduce_add(count);

    if (get_sub_group_local_id() == 0)
        atomic_add(&(output[1]), count);

    if (gdim0 == 0 && gdim1 == 0 && gdim2 == 0) {
        output[0] = OV_INPUT_RANK;
        output[2] = 1;
        output[3] = 1;
    }
}

#undef INPUT0_GET_INDEX1
#undef INPUT_ORDER
