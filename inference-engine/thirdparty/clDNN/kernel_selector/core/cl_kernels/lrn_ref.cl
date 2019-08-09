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
#include "include/fetch.cl"
#include "include/data_types.cl"


KERNEL(normalization)(__global const INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint b = get_global_id(GWS_BATCH);
    const uint f = get_global_id(GWS_FEATURE);
    const uint y = get_global_id(GWS_YX) / INPUT0_SIZE_X;
    const uint x = get_global_id(GWS_YX) % INPUT0_SIZE_X;

    const uint input_index  = GET_DATA_INDEX(INPUT0, b, f, y, x);
    const uint output_index = GET_DATA_INDEX(OUTPUT, b, f, y, x);

    ACCUMULATOR_TYPE sum = 0.0f;
#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

#ifdef ACROSS_CHANNEL

    uint j_offset = input_index - PADDING*INPUT0_FEATURE_PITCH;

    for(int j = 0 ; j < LOCAL_SIZE ; j++)
    {
        const int z_idx = (j + f - PADDING);
        bool zero = (z_idx < 0 || z_idx >= INPUT0_FEATURE_NUM);
        UNIT_TYPE val = zero ? 0.0f : input[j_offset];
        sum += val*val;
        j_offset += INPUT0_FEATURE_PITCH;
#ifdef DYNAMIC_KERNEL_DIVIDER 
        num_elementes += zero ? 0 : 1;
#endif
    }
    
#else

    const int x_start = ((int)x - PADDING);
    const int y_start = ((int)y - PADDING);
    int input_offset = GET_DATA_INDEX(INPUT0, b, f, y_start, x_start);

    for (int j = 0; j < LOCAL_SIZE ; ++j) 
    {
        for (int i = 0; i < LOCAL_SIZE ; ++i) 
        {
            int input_offset_x = x_start + i;
            int input_offset_y = y_start + j;
            bool zero = false;
            zero = input_offset_x < 0 ? true : zero;
            zero = input_offset_y < 0 ? true : zero;
            zero = input_offset_x >= INPUT0_SIZE_X ? true : zero;
            zero = input_offset_y >= INPUT0_SIZE_Y ? true : zero;

            UNIT_TYPE val = zero ? UNIT_VAL_ZERO : input[input_offset];
            
            sum += val*val;
            input_offset += INPUT0_X_PITCH;
#ifdef DYNAMIC_KERNEL_DIVIDER 
            num_elementes += zero ? 0 : 1;
#endif
        }
        input_offset += INPUT0_Y_PITCH - LOCAL_SIZE*INPUT0_X_PITCH;
    }
#endif

#ifdef DYNAMIC_KERNEL_DIVIDER 
    const UNIT_TYPE num_elementes_div = UNIT_VAL_ONE / TO_UNIT_TYPE(num_elementes);
#else
    const UNIT_TYPE num_elementes_div = NUM_ELEMENTS_DIV;
#endif
    
    const UNIT_TYPE base = TO_UNIT_TYPE(K) + TO_UNIT_TYPE((ACCUMULATOR_TYPE)ALPHA*sum * num_elementes_div);
    const UNIT_TYPE normalization_factor = native_powr(base, TO_UNIT_TYPE(-BETA));
    
    const UNIT_TYPE val = input[input_index];
    const UNIT_TYPE normres =  val*normalization_factor;
    output[output_index] = ACTIVATION(normres, ACTIVATION_PARAMS);
}
