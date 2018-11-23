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

#include "include/common.cl"
#include "include/data_types.cl"


KERNEL (permute_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    uint4 input_indices, output_indices;
    
    input_indices[0] = get_global_id(0);
    input_indices[1] = get_global_id(1);
    input_indices[2] = get_global_id(2) % INPUT0_SIZES[2];
    input_indices[3] = get_global_id(2) / INPUT0_SIZES[2];
    
    output_indices[0] = input_indices[PERMUTE_ORDER[0]];
    output_indices[1] = input_indices[PERMUTE_ORDER[1]];
    output_indices[2] = input_indices[PERMUTE_ORDER[2]];
    output_indices[3] = input_indices[PERMUTE_ORDER[3]];
    
    uint input_offset =  INPUT0_OFFSET +
                         input_indices[0]*INPUT0_PITCHES[0] +
                         input_indices[1]*INPUT0_PITCHES[1] +
                         input_indices[2]*INPUT0_PITCHES[2] +
                         input_indices[3]*INPUT0_PITCHES[3];
    uint output_offset = OUTPUT_OFFSET +
                         output_indices[0]*OUTPUT_PITCHES[0] +
                         output_indices[1]*OUTPUT_PITCHES[1] +
                         output_indices[2]*OUTPUT_PITCHES[2] +
                         output_indices[3]*OUTPUT_PITCHES[3];

    output[output_offset] = ACTIVATION(input[input_offset], NL_M, NL_N);
}
