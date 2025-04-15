// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if OUTPUT_DIMS == 5
    #define INPUT_0 input0[INPUT0_GET_INDEX_SAFE(b, f, z, y, x)]
    #if INPUT1_DIMS == 4
        #define INPUT_1 input1[INPUT1_GET_INDEX_SAFE(b, f, y, x)]
    #else
        #define INPUT_1 input1[INPUT1_GET_INDEX_SAFE(b, f, z, y, x)]
    #endif
    #if INPUT2_DIMS == 4
        #define INPUT_2 input2[INPUT2_GET_INDEX_SAFE(b, f, y, x)]
    #else
        #define INPUT_2 input2[INPUT2_GET_INDEX_SAFE(b, f, z, y, x)]
    #endif
#elif OUTPUT_DIMS == 4
    #define INPUT_0 input0[INPUT0_GET_INDEX_SAFE(b, f, y, x)]
    #define INPUT_1 input1[INPUT1_GET_INDEX_SAFE(b, f, y, x)]
    #define INPUT_2 input2[INPUT2_GET_INDEX_SAFE(b, f, y, x)]
#endif

KERNEL(select)(
    OPTIONAL_SHAPE_INFO_ARG
    INPUTS_DECLS
    __global OUTPUT_TYPE* output)
{
    const uint x = (uint)get_global_id(0);
#if OUTPUT_DIMS == 5
    const uint yz  = (uint)get_global_id(1);
    const uint y = yz % OUTPUT_SIZE_Y;
    const uint z = yz / OUTPUT_SIZE_Y;
#elif OUTPUT_DIMS == 4
    const uint y  = (uint)get_global_id(1);
#endif
    const uint bf = (uint)get_global_id(2);
    const uint b = bf % OUTPUT_BATCH_NUM;
    const uint f = bf / OUTPUT_BATCH_NUM;

#if OUTPUT_DIMS == 5
    uint output_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
#elif OUTPUT_DIMS == 4
    uint output_offset = OUTPUT_GET_INDEX(b, f, y, x);
#endif

    #if INPUT1_IS_FP && !OUTPUT_IS_FP
     const OUTPUT_TYPE res = TO_OUTPUT_TYPE(convert_long(select(INPUT_2, INPUT_1, MASK)));
    #else
     const OUTPUT_TYPE res = TO_OUTPUT_TYPE(select(INPUT_2, INPUT_1, MASK));
    #endif

    output[output_offset] = res;
}
