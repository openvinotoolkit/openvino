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

#include "include/include_all.cl"


KERNEL(index_select_gpu_ref)(
    const __global UNIT_TYPE* input,
#ifndef REVERSE
    const __global int* indices,
#endif
    __global UNIT_TYPE* output)
{
    // [CONSTEXPR]:
    const uint input_sx  = INPUT0_SIZE_X;
    const uint input_sy  = INPUT0_SIZE_Y;
    const uint input_sf  = INPUT0_FEATURE_NUM;
    const uint input_sb  = INPUT0_BATCH_NUM;

    const uint out_b         = (uint) get_global_id(0);
    const uint indices_idx   = (uint) get_global_id(1);
    const uint feature_idx   = (uint) get_global_id(2);

    #if AXES_NUMBER == 1
        #ifdef REVERSE
            const uint indices_value = REVERSE_AXIS_SIZE - 1 - indices_idx;
        #else
            const uint indices_value = indices[indices_idx];
        #endif
    #elif AXES_NUMBER > 1
        #ifdef REVERSE
            uint indices_value[4] =  { 
                #ifdef REVERSE_INDEX_SELECT_AXIS_BATCH_SIZE
                    REVERSE_INDEX_SELECT_AXIS_BATCH_SIZE - 1 - out_b,
                #else
                    out_b,
                #endif
                #ifdef REVERSE_INDEX_SELECT_AXIS_FEATURE_SIZE
                    REVERSE_INDEX_SELECT_AXIS_FEATURE_SIZE - 1 - feature_idx,
                #else
                    feature_idx,
                #endif
                #ifdef REVERSE_INDEX_SELECT_AXIS_Y_SIZE
                    REVERSE_INDEX_SELECT_AXIS_Y_SIZE - 1 - indices_idx,
                #else
                    indices_idx,
                #endif             
                    0     
             };
        #endif
    #endif
    
    // [LOGIC]:
    #if AXES_NUMBER > 1
        for(uint x = 0; x < input_sx; x++)
        {
            #ifdef REVERSE_INDEX_SELECT_AXIS_X_SIZE
                indices_value[3] = REVERSE_INDEX_SELECT_AXIS_X_SIZE - 1 - x;
            #else
                indices_value[3] = x;
            #endif
            output[GET_DATA_INDEX(OUTPUT, out_b, feature_idx, indices_idx, x)] = input[GET_DATA_INDEX(INPUT0, indices_value[0], indices_value[1], indices_value[2], indices_value[3])];
        }
            
    #else
        #ifdef INDEX_SELECT_AXIS_BATCH
            for(uint x = 0; x < input_sx; x++)
            { 
                for(uint y = 0; y < input_sy; y++)
                {  
                    output[GET_DATA_INDEX(OUTPUT, indices_idx, feature_idx, y, x)] = input[GET_DATA_INDEX(INPUT0, indices_value, feature_idx, y, x)];
                }
            }
        #elif defined INDEX_SELECT_AXIS_FEATURE
            for(uint x = 0; x < input_sx; x++)
            {
                output[GET_DATA_INDEX(OUTPUT, out_b, indices_idx, feature_idx, x)] = input[GET_DATA_INDEX(INPUT0, out_b, indices_value, feature_idx, x)];
            }
        #elif defined INDEX_SELECT_AXIS_X
            for(uint i = 0; i < input_sy; i++)
            {
                output[GET_DATA_INDEX(OUTPUT, out_b, feature_idx, i, indices_idx)] = input[GET_DATA_INDEX(INPUT0, out_b, feature_idx, i, indices_value)];
            }
        #elif defined INDEX_SELECT_AXIS_Y

            for(uint i = 0; i < input_sx; i++)
            {
                output[GET_DATA_INDEX(OUTPUT, out_b, feature_idx, indices_idx, i)] = input[GET_DATA_INDEX(INPUT0, out_b, feature_idx, indices_value, i)];
            }
        #endif
    #endif
}