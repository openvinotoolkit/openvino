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

// change this function with your own idea. please note that it's a naive implementation.
KERNEL(convolution_tutorial)(
    __global INPUT0_TYPE* input,        // input buffer
    __global OUTPUT_TYPE* output,       // output buffer
    __global FILTER_TYPE* weights,      // weights buffer (training output)
#if BIAS_TERM                           // in case we have bias in convolution params
    __global BIAS_TYPE* biases,         // bias buffer (training output)
#endif
    uint split_idx)                     // which split index to process
{
#if defined OUTPUT_LAYOUT_YXFB                  // in Case of YXFB we need a different processing order than BFYX (from performance aspect)
    const uint x = get_global_id(1);
    const uint y = get_global_id(2);
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(0);
    const uint b = 0;
#else
    const uint f = get_global_id(0) % OUTPUT_FEATURE_NUM;
    const uint b = get_global_id(0) / OUTPUT_FEATURE_NUM;
#endif
#else
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
#if OUTPUT_BATCH_NUM == 1
    const uint f = get_global_id(2);
    const uint b = 0;
#else
    const uint f = get_global_id(2) % OUTPUT_FEATURE_NUM;
    const uint b = get_global_id(2) / OUTPUT_FEATURE_NUM;
#endif
#endif

    UNIT_TYPE dotProd = UNIT_VAL_ZERO;                                          // UNIT_TYPE - half/float/etc
    
#if BIAS_TERM
    #if   BIAS_PER_OUTPUT
        const uint bias_index = GET_DATA_INDEX(BIAS, b, f, y, x);               // helper macro to cacluate indices
    #elif BIAS_PER_OFM
        const uint bias_index = f;
    #endif
    dotProd = biases[bias_index];
#endif

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    // in case of depth separable optimization we have to dynamically calculate the split index
#if DEPTHWISE_SEPARABLE_OPT
    const uint in_split_offset = (f / FILTER_OFM_NUM) * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#else
    const uint in_split_offset = split_idx * INPUT0_FEATURE_PITCH * FILTER_IFM_NUM;
#endif
    const uint filter_offset = f*FILTER_OFM_PITCH;
    const uint input_offset = b*INPUT0_BATCH_PITCH + INPUT0_OFFSET + in_split_offset;

    for (uint k = 0; k < FILTER_IFM_NUM; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;
            const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

            if(!zero_y)
            {
                for (uint i = 0; i < FILTER_SIZE_X ; ++i)
                {
                    const int input_offset_x = input_x + i * DILATION_SIZE_X;
                    const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                    if(!zero_x)
                    {
                        uint input_idx = input_offset + (uint)input_offset_x*INPUT0_X_PITCH + (uint)input_offset_y*INPUT0_Y_PITCH + k*INPUT0_FEATURE_PITCH;
                        uint filter_idx = filter_offset + k*FILTER_IFM_PITCH + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;
                        dotProd += input[input_idx]*weights[filter_idx];    // finally the convolution calcualtion.
                    }
                }
            }
        }
    }
    
    const uint out_split_offset = split_idx * OUTPUT_FEATURE_PITCH * OUTPUT_FEATURE_NUM;    // calculating output split offset
    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, f, y, x) + out_split_offset;           // helper macro to calculate output index
    output[dst_index] = ACTIVATION(dotProd, NL_M, NL_N);                                    // run activation functions (RelU in most cases) and set output
}

#else

//#include "put here your include files"

__kernel void convolution_tutorial(
    const __global UNIT_TYPE* input,
    __global UNIT_TYPE* output,
    const __global UNIT_TYPE* filter,
    const __global UNIT_TYPE* bias)
{
    // fill here your kernel
}

#endif