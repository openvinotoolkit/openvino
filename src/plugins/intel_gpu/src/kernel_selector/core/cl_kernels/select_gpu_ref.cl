// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#define GET_INDEX(prefix, num)                                                      \
    CAT(CAT(prefix, num), _OFFSET) +                                                \
    (d1 % CAT(CAT(prefix, num), _SIZES)[0])*CAT(CAT(prefix, num), _PITCHES)[0] +    \
    (d2 % CAT(CAT(prefix, num), _SIZES)[1])*CAT(CAT(prefix, num), _PITCHES)[1] +    \
    (d3 % CAT(CAT(prefix, num), _SIZES)[2])*CAT(CAT(prefix, num), _PITCHES)[2] +    \
    (d4 % CAT(CAT(prefix, num), _SIZES)[3])*CAT(CAT(prefix, num), _PITCHES)[3]

#define INPUT_0 input0[GET_INDEX(INPUT, 0)]
#define INPUT_1 input1[GET_INDEX(INPUT, 1)]
#define INPUT_2 input2[GET_INDEX(INPUT, 2)]

KERNEL(select)(
    INPUTS_DECLS
    __global OUTPUT_TYPE* output)
{

const uint d1  = (uint) get_global_id(0);
const uint d2  = (uint) get_global_id(1);
const uint d34 = (uint) get_global_id(2);

const uint d3  = d34 % OUTPUT_SIZES[2];
const uint d4  = d34 / OUTPUT_SIZES[2];

uint output_offset = OUTPUT_OFFSET +
                     d1*OUTPUT_PITCHES[0] +
                     d2*OUTPUT_PITCHES[1] +
                     d3*OUTPUT_PITCHES[2] +
                     d4*OUTPUT_PITCHES[3];

const OUTPUT_TYPE res = select(INPUT_2, INPUT_1, MASK);

output[output_offset] = res;
}
