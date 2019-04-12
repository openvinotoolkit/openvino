// Copyright (c) 2017 Intel Corporation
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


KERNEL (permute_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    uint4 input_indices, output_indices;
    
    //gws(y, x, b*f)
    //input_indices[b, f, x, y]
    input_indices[3] = get_global_id(0); 
    input_indices[2] = get_global_id(1);
    input_indices[1] = get_global_id(2) % INPUT0_FEATURE_NUM;
    input_indices[0] = get_global_id(2) / INPUT0_FEATURE_NUM;
    
    //PERMUTE_ORDER[b, f, x, y]
    //output_indices[b, f, x, y]
    output_indices[0] = input_indices[PERMUTE_ORDER[0]];
    output_indices[1] = input_indices[PERMUTE_ORDER[1]];
    output_indices[2] = input_indices[PERMUTE_ORDER[2]];
    output_indices[3] = input_indices[PERMUTE_ORDER[3]];
    
    uint input_offset =  GET_DATA_INDEX(INPUT0, input_indices[0], input_indices[1], input_indices[3], input_indices[2]);
    uint output_offset = GET_DATA_INDEX(OUTPUT, output_indices[0], output_indices[1], output_indices[3], output_indices[2]);

    output[output_offset] = ACTIVATION(input[input_offset], NL_M, NL_N);
}
