// Copyright (c) 2016-2017 Intel Corporation
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

#ifdef ADVANCED_TUTORIAL

#include "include/include_all.cl"

KERNEL(activation)(
    __global UNIT_TYPE* input, 
    __global UNIT_TYPE* output
#ifdef PARAMETERIZED 
    , __global ADDITIONAL_PARAMS_TYPE* params
#endif
    )
{
#if defined OUTPUT_LAYOUT_YXFB                  // in Case of YXFB we need a different processing order than BFYX (from performance aspect)
    const uint x = get_global_id(1);
    const uint y = get_global_id(2);
#if OUTPUT_BATCH_NUM == 1
    const uint feature = get_global_id(0);
    const uint batch = 0;
#else
    const uint feature = get_global_id(0) % OUTPUT_FEATURE_NUM;
    const uint batch = get_global_id(0) / OUTPUT_FEATURE_NUM;
#endif
#else
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const uint feature = get_global_id(2);
    const uint batch = 0;
#else
    const uint feature = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint batch = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
#endif

    const uint src_index = GET_DATA_INDEX(INPUT0, batch, feature, y, x);    // helper macro to deduce the buffer index.
    const uint dst_index = GET_DATA_INDEX(OUTPUT, batch, feature, y, x);

#if defined PARAMETERIZED                                                   // in case that the input additional params is located on a bufffer
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
    output[dst_index] = ACTIVATION(input[src_index], nl_m, nl_n);           // Do the activation
}

#else

//#include "put here your include files"

__kernel void activation_tutorial(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output)
{
	// fill here your kernel
}

#endif