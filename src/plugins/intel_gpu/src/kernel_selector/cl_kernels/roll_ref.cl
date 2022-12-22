// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define GET_INDEX(prefix, ORDER) CAT(prefix, _GET_INDEX)(ORDER)

KERNEL(roll_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

#if OUTPUT_DIMS == 4
#    define ORDER b, f, y, x
    uint x = dim0;
    uint y = dim1;
#elif OUTPUT_DIMS == 5
#    define ORDER b, f, z, y, x
    uint x = dim0 % OUTPUT_SIZE_X;
    uint y = dim0 / OUTPUT_SIZE_X;
    uint z = dim1;
#elif OUTPUT_DIMS == 6
#    define ORDER b, f, w, z, y, x
    uint x = dim0 % OUTPUT_SIZE_X;
    uint y = dim0 / OUTPUT_SIZE_X;
    uint z = dim1 % OUTPUT_SIZE_Z;
    uint w = dim1 / OUTPUT_SIZE_Z;
#endif
    uint f = dim2 % OUTPUT_FEATURE_NUM;
    uint b = dim2 / OUTPUT_FEATURE_NUM;

    const uint input_index = GET_INDEX(INPUT0, ORDER);

#if OUTPUT_DIMS >= 4
    x = (x + SHIFT_SIZE_X) % OUTPUT_SIZE_X;
    y = (y + SHIFT_SIZE_Y) % OUTPUT_SIZE_Y;
#endif
#if OUTPUT_DIMS >= 5
    z = (z + SHIFT_SIZE_Z) % OUTPUT_SIZE_Z;
#endif
#if OUTPUT_DIMS == 6
    w = (w + SHIFT_SIZE_W) % OUTPUT_SIZE_W;
#endif
    f = (f + SHIFT_FEATURE_NUM) % OUTPUT_FEATURE_NUM;
    b = (b + SHIFT_BATCH_NUM) % OUTPUT_BATCH_NUM;

    const uint output_index = GET_INDEX(OUTPUT, ORDER);

    output[output_index] = input[input_index];
}

#undef GET_INDEX
#undef ORDER
