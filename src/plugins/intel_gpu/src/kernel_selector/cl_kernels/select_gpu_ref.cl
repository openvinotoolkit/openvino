// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define INPUT_0 input0[INPUT0_GET_INDEX_SAFE(b, f, y, x)]
#define INPUT_1 input1[INPUT1_GET_INDEX_SAFE(b, f, y, x)]
#define INPUT_2 input2[INPUT2_GET_INDEX_SAFE(b, f, y, x)]

KERNEL(select)(
    OPTIONAL_SHAPE_INFO_ARG
    INPUTS_DECLS
    __global OUTPUT_TYPE* output)
{
    const uint x  = (uint)get_global_id(0);
    const uint y  = (uint)get_global_id(1);
    const uint bf = (uint)get_global_id(2);

    const uint b = bf % OUTPUT_BATCH_NUM;
    const uint f = bf / OUTPUT_BATCH_NUM;

    uint output_offset = OUTPUT_GET_INDEX(b, f, y, x);

    const OUTPUT_TYPE res = select(INPUT_2, INPUT_1, MASK);

    output[output_offset] = res;
}
