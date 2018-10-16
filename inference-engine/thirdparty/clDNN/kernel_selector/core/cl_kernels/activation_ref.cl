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

#include "include/include_all.cl"

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
#if defined OUTPUT_LAYOUT_YXFB
    const unsigned x = get_global_id(1);
    const unsigned y = get_global_id(2);
#if OUTPUT_BATCH_NUM == 1
    const unsigned feature = get_global_id(0);
    const unsigned batch = 0;
#else
    const unsigned feature = get_global_id(0) % OUTPUT_FEATURE_NUM;
    const unsigned batch = get_global_id(0) / OUTPUT_FEATURE_NUM;
#endif
#else
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

#if GRADIENT
    const unsigned src_grad_index = batch*INPUT0_BATCH_PITCH + feature*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH + INPUT0_OFFSET;
    const unsigned src_index = batch*INPUT1_BATCH_PITCH + feature*INPUT1_FEATURE_PITCH + y*INPUT1_Y_PITCH + x*INPUT1_X_PITCH + INPUT1_OFFSET;
#else
    const unsigned src_index = batch*INPUT0_BATCH_PITCH + feature*INPUT0_FEATURE_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH + INPUT0_OFFSET;
#endif
    const unsigned dst_index = batch*OUTPUT_BATCH_PITCH + feature*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH + OUTPUT_OFFSET;

#if defined PARAMETERIZED
    #if   PARAMS_NUM == 2
        const float nl_m = (float)params[2*feature + 0];
        const float nl_n = (float)params[2*feature + 1];
    #elif PARAMS_NUM == 1
        const float nl_m = (float)params[feature];
        const float nl_n = (float)NL_N;
    #else
        const float nl_m = (float)NL_M;
        const float nl_n = (float)NL_N;
    #endif
#else
    const float nl_m = (float)NL_M;
    const float nl_n = (float)NL_N;
#endif

#if GRADIENT
    output_grad[dst_index] = ACTIVATION(input_grad[src_grad_index], input[src_index], nl_m, nl_n);
#else
    output[dst_index] = ACTIVATION(input[src_index], nl_m, nl_n);
#endif
}