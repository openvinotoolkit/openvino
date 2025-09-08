// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define BATCH_INDEX 0
#define FEATURE_INDEX 1
#if INPUT0_DIMS == 4
    #define Y_INDEX 2
    #define X_INDEX 3
#elif INPUT0_DIMS == 5
    #define Z_INDEX 2
    #define Y_INDEX 3
    #define X_INDEX 4
#elif INPUT0_DIMS == 6
    #define W_INDEX 2
    #define Z_INDEX 3
    #define Y_INDEX 4
    #define X_INDEX 5
#endif


KERNEL(reverse_ref)(
        const __global INPUT0_TYPE* input,
        const __global INPUT1_TYPE* axis,
        __global OUTPUT_TYPE* output)
{
    uint x = get_global_id(0);
    uint b = (uint)get_global_id(2) % INPUT0_BATCH_NUM;
    uint f = (uint)get_global_id(2) / INPUT0_BATCH_NUM;
#if INPUT0_DIMS == 4
    //|dim2:bf|dim1:y|dim0:x
    uint y = get_global_id(1);
    const uint input_index = INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    //|dim2:bf|dim1:yz|dim0:x
    uint z = get_global_id(1) / INPUT0_SIZE_Y;
    uint y = get_global_id(1) % INPUT0_SIZE_Y;
    const uint input_index = INPUT0_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    //|dim2:bf|dim1:wyz|dim0:x
    uint y = get_global_id(1) % INPUT0_SIZE_Y;
    uint z = get_global_id(1) / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    uint w = get_global_id(1) / (INPUT0_SIZE_Y * INPUT0_SIZE_Z) % INPUT0_SIZE_W;
    const uint input_index = INPUT0_GET_INDEX(b, f, w, z, y, x);
#endif //INPUT0_DIMS

#ifdef MASK_MODE
    if (axis[BATCH_INDEX]) {
        b = OUTPUT_BATCH_NUM - b - 1;
    }
    if (axis[FEATURE_INDEX]) {
        f = OUTPUT_FEATURE_NUM - f - 1;
    }
    if (axis[Y_INDEX]) {
        y = OUTPUT_SIZE_Y - y - 1;
    }
    if (axis[X_INDEX]) {
        x = OUTPUT_SIZE_X - x - 1;
    }
#if INPUT0_DIMS == 4
    const uint output_index =  OUTPUT_GET_INDEX(b, f, y, x);
#else
    if (axis[Z_INDEX]) {
        z = OUTPUT_SIZE_Z - z - 1;
    }
#if INPUT0_DIMS == 5
    const uint output_index = OUTPUT_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    if (axis[W_INDEX]) {
        w = OUTPUT_SIZE_W - w - 1;
    }
    const uint output_index = OUTPUT_GET_INDEX(b, f, w, z, y, x);
#endif
#endif
#else  //reverse mode

    for (uint i = 0; i < INPUT1_FEATURE_NUM; ++i) {
        if (axis[i] == BATCH_INDEX) {
            b = OUTPUT_BATCH_NUM - b - 1;
        } else if (axis[i] == FEATURE_INDEX) {
            f = OUTPUT_FEATURE_NUM - f - 1;
        } else  if (axis[i] == Y_INDEX) {
            y = OUTPUT_SIZE_Y - y - 1;
        } else if (axis[i] == X_INDEX ) {
            x = OUTPUT_SIZE_X -x - 1;
        }
#if INPUT0_DIMS >= 5
        else if (axis[i] == Z_INDEX) {
            z = OUTPUT_SIZE_Z - z - 1;
        }
#if INPUT0_DIMS == 6
        else if (axis[i] == W_INDEX) {
            w = OUTPUT_SIZE_W - w - 1;
        }
#endif // INPUT0_DIMS == 6
#endif //INPUT0_DIMS >= 5
    }
#if INPUT0_DIMS == 4
    const uint output_index =  OUTPUT_GET_INDEX(b, f, y, x);
#elif INPUT0_DIMS == 5
    const uint output_index =  OUTPUT_GET_INDEX(b, f, z, y, x);
#elif INPUT0_DIMS == 6
    const uint output_index =  OUTPUT_GET_INDEX(b, f, w, z, y, x);
#endif // INPUT0_DIMS

#endif // reverse mode
    output[output_index] = input[input_index];
}

#undef X_INDEX
#undef Y_INDEX
#undef Z_INDEX
#undef W_INDEX
