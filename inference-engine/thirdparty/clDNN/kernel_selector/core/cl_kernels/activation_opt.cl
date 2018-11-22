/*
// Copyright (c) 2016 Intel Corporation
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

#include "include/common.cl"
#include "include/data_types.cl"

KERNEL(activation)(
#if GRADIENT
    __global UNIT_TYPE* input_grad,
    __global UNIT_TYPE* output,
    __global UNIT_TYPE* input
#else
    __global UNIT_TYPE* input, 
    __global UNIT_TYPE* output
#endif
    )
{
    const unsigned int x = get_global_id(0) * NUM_COLS_WI;

    unsigned int input_offset  = x + INPUT0_OFFSET; 
    unsigned int output_offset = x + OUTPUT_OFFSET; 

    typedef CAT(UNIT_TYPE, 4) type_t;
#if GRADIENT
    type_t g = ((__global type_t*) (input_grad + input_offset))[0];
#endif
    type_t v = ((__global type_t*) (input + input_offset))[0];

#if GRADIENT
    v = ACTIVATION(g, v, NL_M, NL_N);
#else
    v = ACTIVATION(v, NL_M, NL_N);
#endif

    *((__global type_t*)(output + output_offset)) = v;
}
