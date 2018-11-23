/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "include/include_all.cl"

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

const OUTPUT_TYPE res = select(INPUT_1, INPUT_0, MASK);

output[output_offset] = res;
}
