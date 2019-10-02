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
#include "include/fetch.cl"

#ifdef PARAMETERIZED
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#else
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

// TODO: move it from layout based to memory based
KERNEL(activation)(
#if GRADIENT
    __global UNIT_TYPE* input_grad,
    __global UNIT_TYPE* output_grad,
    __global UNIT_TYPE* input
#else
    __global UNIT_TYPE* input,
    __global UNIT_TYPE* output
#endif
#ifdef PARAMETERIZED
    , __global ADDITIONAL_PARAMS_TYPE* params
#endif
    )
{
#if OUTPUT_DIMS == 5
    #define ORDER batch,feature,z,y,x
#elif OUTPUT_DIMS == 4
    #define ORDER batch,feature,y,x
#endif

#if defined OUTPUT_LAYOUT_BFZYX
    const unsigned x = get_global_id(0);
    const uint y = get_global_id(1) % OUTPUT_SIZE_Y;
    const uint z = get_global_id(1) / OUTPUT_SIZE_Y;
#if OUTPUT_BATCH_NUM == 1
    const unsigned feature = get_global_id(2);
    const unsigned batch = 0;
#else
    const unsigned feature = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const unsigned batch = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
#else
#if defined OUTPUT_LAYOUT_YXFB || defined OUTPUT_LAYOUT_BFYX_F16
    const unsigned x = get_global_id(1);
    const unsigned y = get_global_id(2);
#define z 0
#if OUTPUT_BATCH_NUM == 1
    const unsigned feature = get_global_id(0);
    const unsigned batch = 0;
#else
    const unsigned feature = get_global_id(0) % OUTPUT_FEATURE_NUM;
    const unsigned batch = get_global_id(0) / OUTPUT_FEATURE_NUM;
#endif
#else
#define z 0
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const unsigned feature = get_global_id(2);
    const unsigned batch = 0;
#else
    const unsigned feature = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const unsigned batch = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
#endif
#endif

#if GRADIENT
    const unsigned src_grad_index = GET_INDEX(INPUT,0,ORDER);
    const unsigned src_index = GET_INDEX(INPUT,1,ORDER);
#else
    const unsigned src_index = GET_INDEX(INPUT,0,ORDER);
#endif
    const unsigned dst_index = GET_INDEX(OUTPUT,,ORDER);

#if defined PARAMETERIZED
    #if PARAMS_NUM > 2
        #error Too many params
    #elif PARAMS_NUM == 2
        #define NL_M_PARAMETERIZED (float)params[2*feature + 0]
        #define NL_N_PARAMETERIZED (float)params[2*feature + 1]
    #elif PARAMS_NUM == 1
        #define NL_M_PARAMETERIZED (float)params[feature]
        #define NL_N_PARAMETERIZED (float)NL_N
    #else
        #define NL_M_PARAMETERIZED (float)NL_M
        #define NL_N_PARAMETERIZED (float)NL_N
    #endif
    #define PARAMETERIZED_ACTIVATION_PARAMS NL_M_PARAMETERIZED, NL_N_PARAMETERIZED

    #if GRADIENT
        output_grad[dst_index] = ACTIVATION(input_grad[src_grad_index], input[src_index], PARAMETERIZED_ACTIVATION_PARAMS);
    #else
        output[dst_index] = ACTIVATION(input[src_index], PARAMETERIZED_ACTIVATION_PARAMS);
    #endif
#else
    #if GRADIENT
        output_grad[dst_index] = ACTIVATION(input_grad[src_grad_index], input[src_index], ACTIVATION_PARAMS);
    #else
        output[dst_index] = ACTIVATION(input[src_index], ACTIVATION_PARAMS);
    #endif
#endif
}
