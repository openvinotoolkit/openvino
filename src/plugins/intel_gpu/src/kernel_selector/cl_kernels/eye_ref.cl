// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(eye_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output) {
    // Get the index of the current element to be processed
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
#if OUTPUT_DIMS <= 4
    const uint xy = get_global_id(2);
    const uint y = xy / OUTPUT_SIZE_X;
    const uint x = xy % OUTPUT_SIZE_X;
    const uint idx = OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_DIMS == 5
    const uint xyz = get_global_id(2);
    const uint yx = xyz % (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint z = xyz / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
    const uint y = yx / OUTPUT_SIZE_X;
    const uint x = yx % OUTPUT_SIZE_X;
    const uint idx = OUTPUT_GET_INDEX(b, f, z, y, x);
#endif
    if (x - DIAGONAL == y) {
        output[idx] = 1;
    } else {
        output[idx] = 0;
    }
}
