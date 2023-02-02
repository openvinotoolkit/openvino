// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#ifdef IS_DYNAMIC
    #define GET_INDEX(prefix) GET_DATA_INDEX_SAFE(prefix, d4, d3, d2, d1)
#else
    #define GET_INDEX(prefix)                                       \
        CAT(prefix, _OFFSET) +                                      \
        (d1 % CAT(prefix, _SIZES)[0])*CAT(prefix, _PITCHES)[0] +    \
        (d2 % CAT(prefix, _SIZES)[1])*CAT(prefix, _PITCHES)[1] +    \
        (d3 % CAT(prefix, _SIZES)[2])*CAT(prefix, _PITCHES)[2] +    \
        (d4 % CAT(prefix, _SIZES)[3])*CAT(prefix, _PITCHES)[3]
#endif

#define INPUT_0 input0[GET_INDEX(INPUT0)]
#define INPUT_1 input1[GET_INDEX(INPUT1)]
#define INPUT_2 input2[GET_INDEX(INPUT2)]

KERNEL(select)(
    OPTIONAL_SHAPE_INFO_ARG
    INPUTS_DECLS
    __global OUTPUT_TYPE* output)
{

const uint d1  = (uint) get_global_id(0);
const uint d2  = (uint) get_global_id(1);
const uint d34 = (uint) get_global_id(2);

#ifdef IS_DYNAMIC
    const uint d3 = d34 % OUTPUT_FEATURE_NUM;
    const uint d4 = d34 / OUTPUT_FEATURE_NUM;
#else
    const uint d3 = d34 % OUTPUT_SIZES[2];
    const uint d4 = d34 / OUTPUT_SIZES[2];
#endif

#ifdef IS_DYNAMIC
    uint output_offset = OUTPUT_GET_INDEX(d4, d3, d2, d1);
#else
    uint output_offset = GET_DATA_INDEX_RAW(OUTPUT, d1, d2, d3, d4);
#endif

const OUTPUT_TYPE res = select(INPUT_2, INPUT_1, MASK);

output[output_offset] = res;
}
