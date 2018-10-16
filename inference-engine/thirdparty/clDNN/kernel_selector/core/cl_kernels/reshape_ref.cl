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

KERNEL (reshape_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint d1 = get_global_id(0);
    const uint d2 = get_global_id(1);
    const uint d3 = get_global_id(2) % INPUT0_SIZES[2];
    const uint d4 = get_global_id(2) / INPUT0_SIZES[2];

    uint linear = d1 + d2*INPUT0_SIZES[0] + d3*INPUT0_SIZES[0]*INPUT0_SIZES[1] + d4*INPUT0_SIZES[0]*INPUT0_SIZES[1]*INPUT0_SIZES[2];

    const uint od1 = linear % OUTPUT_SIZES[0]; linear /= OUTPUT_SIZES[0];
    const uint od2 = linear % OUTPUT_SIZES[1]; linear /= OUTPUT_SIZES[1];
    const uint od3 = linear % OUTPUT_SIZES[2]; linear /= OUTPUT_SIZES[2];
    const uint od4 = linear;
    
    uint input_offset =  INPUT0_OFFSET +
                         d1*INPUT0_PITCHES[0] +
                         d2*INPUT0_PITCHES[1] +
                         d3*INPUT0_PITCHES[2] +
                         d4*INPUT0_PITCHES[3];
    uint output_offset = OUTPUT_OFFSET +
                         od1*OUTPUT_PITCHES[0] +
                         od2*OUTPUT_PITCHES[1] +
                         od3*OUTPUT_PITCHES[2] +
                         od4*OUTPUT_PITCHES[3];
    
    output[output_offset] = ACTIVATION(input[input_offset], NL_M ,NL_N);
}
