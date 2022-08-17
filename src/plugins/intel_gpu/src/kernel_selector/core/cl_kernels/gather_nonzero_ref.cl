// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"

#define INPUT0_GET_INDEX1(idx_order) INPUT0_GET_INDEX(idx_order)

KERNEL (gather_nonzero_ref)(const __global INPUT0_TYPE* input,
                            volatile __global INPUT1_TYPE* output_shape,
                            __global OUTPUT_TYPE* output)
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

    int num_nonzero_acc = (input[INPUT0_GET_INDEX1(INPUT_ORDER)] == INPUT0_VAL_ZERO) ? 0 : 1;
    num_nonzero_acc = sub_group_scan_inclusive_add(num_nonzero_acc);
    
    int pos;

    if (get_sub_group_local_id() == (get_sub_group_size() - 1)) {
        pos = atomic_add(&(output_shape[2]), num_nonzero_acc);
        pos = pos - 1;
    }

    pos = sub_group_broadcast(pos, (get_sub_group_size() - 1));
  
    // output_shape = {rank, # nonzero, 1, 1}
    if (input[INPUT0_GET_INDEX1(INPUT_ORDER)] != INPUT0_VAL_ZERO) {
        const int num_nonzero = output_shape[1];

        pos = pos + num_nonzero_acc - 1;

        int output_b = pos;
        int output_f = pos + num_nonzero;

        output[output_b] = b;
        output[output_f] = f;

        #if INPUT0_DIMS == 6
            int output_w = pos + num_nonzero * 2;
            int output_z = pos + num_nonzero * 3;
            int output_y = pos + num_nonzero * 4;
            int output_x = pos + num_nonzero * 5;

            output[output_w] = w;
            output[output_z] = z;
        #elif INPUT0_DIMS == 5
            int output_z = pos + num_nonzero * 2;
            int output_y = pos + num_nonzero * 3;
            int output_x = pos + num_nonzero * 4;

            output[output_z] = z;
        #elif INPUT0_DIMS == 4
            int output_y = pos + num_nonzero * 2;
            int output_x = pos + num_nonzero * 3;
        #endif

        output[output_y] = y;
        output[output_x] = x;
    }
}

#undef INPUT0_GET_INDEX1
#undef INPUT_ORDER
