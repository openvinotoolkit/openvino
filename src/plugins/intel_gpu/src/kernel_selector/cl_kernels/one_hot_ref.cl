// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_LAYOUT_BFZYX
    #define GET_COORDS_INDEX(prefix, coords) GET_DATA_INDEX_5D(prefix, coords[0], coords[1], coords[2], coords[3], coords[4])
#else
    #define GET_COORDS_INDEX(prefix, coords) GET_DATA_INDEX(prefix, coords[0], coords[1], coords[2], coords[3])
#endif

KERNEL(one_hot_ref)(const __global INPUT0_TYPE* input,
                          __global OUTPUT_TYPE* output)
{
#if OUTPUT_LAYOUT_BFZYX && INPUT0_LAYOUT_BFYX
    uint in_coords[5] = { get_global_id(0), (uint)get_global_id(1) / INPUT0_SIZE_Z, (uint)get_global_id(1) % INPUT0_SIZE_Z, (uint)get_global_id(2) / INPUT0_SIZE_X, (uint)get_global_id(2) % INPUT0_SIZE_X };
    uint out_coords[5] = { get_global_id(0), get_global_id(1), (uint)get_global_id(2) / INPUT0_SIZE_X, (uint)get_global_id(2) % INPUT0_SIZE_X, 1 };
    const uint dims_num = 5;
#elif OUTPUT_LAYOUT_BFZYX
    uint in_coords[5] = { get_global_id(0), (uint)get_global_id(1) / INPUT0_SIZE_Z, (uint)get_global_id(1) % INPUT0_SIZE_Z, (uint)get_global_id(2) / INPUT0_SIZE_X, (uint)get_global_id(2) % INPUT0_SIZE_X };
    uint out_coords[5] = { get_global_id(0), (uint)get_global_id(1) / INPUT0_SIZE_Z, (uint)get_global_id(1) % INPUT0_SIZE_Z, (uint)get_global_id(2) / INPUT0_SIZE_X, (uint)get_global_id(2) % INPUT0_SIZE_X };
    const uint dims_num = 5;
#else
    uint in_coords[4] = { get_global_id(0), get_global_id(1), (uint)get_global_id(2) / INPUT0_SIZE_X, (uint)get_global_id(2) % INPUT0_SIZE_X };
    uint out_coords[4] = { get_global_id(0), get_global_id(1), (uint)get_global_id(2) / INPUT0_SIZE_X, (uint)get_global_id(2) % INPUT0_SIZE_X };
    const uint dims_num = 4;
#endif
    for (int i = dims_num - 1; i > ONE_HOT_AXIS; --i)
        out_coords[i] = out_coords[i - 1];

    // Fill the output with 0
    for (out_coords[ONE_HOT_AXIS] = 0; out_coords[ONE_HOT_AXIS] < ONE_HOT_LIMIT; ++out_coords[ONE_HOT_AXIS])
        output[GET_COORDS_INDEX(OUTPUT, out_coords)] = TO_OUTPUT_TYPE(OFF_VALUE);

    // Put in the 1; ignore bad input values
    INPUT0_TYPE val = input[GET_COORDS_INDEX(INPUT0, in_coords)];
    if (val >= 0 && val < ONE_HOT_LIMIT) {
        out_coords[ONE_HOT_AXIS] = val;
        output[GET_COORDS_INDEX(OUTPUT, out_coords)] = TO_OUTPUT_TYPE(ON_VALUE);
    }
}
