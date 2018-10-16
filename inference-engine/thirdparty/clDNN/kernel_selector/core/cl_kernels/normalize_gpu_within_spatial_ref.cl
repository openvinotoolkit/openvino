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


#include "include/include_all.cl"

#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif


KERNEL (normalize_gpu_within_spatial_bfyx)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output, const __global UNIT_TYPE* scale_input)
{
    const uint x = get_global_id(0);
    const uint y = get_global_id(1);
    const uint b = get_global_id(2);

    const uint input_first = INPUT0_OFFSET + b*INPUT0_BATCH_PITCH + y*INPUT0_Y_PITCH + x*INPUT0_X_PITCH;

    // Compute norm
    uint input_idx = input_first;
    float norm = EPSILON;
    for (int i = 0; i < INPUT0_FEATURE_NUM; i++)
    {
        float value = (float)input[input_idx];
        norm = mad(value, value, norm);
        input_idx += INPUT0_FEATURE_PITCH;
    }
   
    uint output_idx = OUTPUT_OFFSET + b*OUTPUT_BATCH_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH;

    if(norm <= THRESHOLD)
    {
        norm = 0;
    }
    else
    {
        norm = native_powr(norm, -0.5f);
    }

    // Scale the input
    input_idx = input_first;
    for (int f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
#if SCALE_TABLE_FEATURE_NUM == 1
        const uint scale_index = 0;
#elif INPUT0_FEATURE_NUM <= SCALE_TABLE_FEATURE_NUM
        const uint scale_index = f;
#else
        const uint scale_index = f % SCALE_TABLE_FEATURE_NUM;
#endif 

        output[output_idx] = ACTIVATION(UNIT_CVT_FUNC(norm) * input[input_idx] * scale_input[scale_index], NL_M, NL_N);
        output_idx += OUTPUT_FEATURE_PITCH;
        input_idx += INPUT0_FEATURE_PITCH;
    }
}


#undef UNIT_CVT_FUNC
