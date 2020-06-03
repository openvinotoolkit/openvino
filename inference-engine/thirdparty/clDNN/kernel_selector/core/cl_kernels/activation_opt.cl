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
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const unsigned int x = (uint)get_global_id(0) * NUM_COLS_WI;
    unsigned int input_offset  = x + INPUT0_OFFSET;
    unsigned int output_offset = x + OUTPUT_OFFSET;

    typedef CAT(INPUT0_TYPE, 4) input_t;
    typedef CAT(OUTPUT_TYPE, 4) output_t;

    input_t v = ((__global input_t*) (input + input_offset))[0];

    v = ACTIVATION_KERNEL(v, ACTIVATION_PARAMS_KERNEL);

#if HAS_FUSED_OPS
    output_t result;
    #if !CAN_USE_VECTOR
        for (int i = 0; i < 4; i++) {
            FUSED_OPS_SCALAR;
            result[i] = FUSED_OPS_RESULT_SCALAR;
        }
    #else
        FUSED_OPS_VECTOR;
        result = FUSED_OPS_RESULT_VECTOR;
    #endif
    *((__global output_t*)(output + output_offset)) = result;
#else
    *((__global output_t*)(output + output_offset)) = v;
#endif
}
