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

KERNEL (reorder_weights_image_2d_c4_fyx_b)(const __global INPUT0_TYPE* input, write_only image2d_t output)
{
    const unsigned o = get_global_id(0);
    const unsigned iyx = get_global_id(1);
    const unsigned x = iyx % INPUT0_SIZE_X;
    const unsigned y = (iyx / INPUT0_SIZE_X) % INPUT0_SIZE_Y;
    const unsigned i = y / INPUT0_SIZE_Y;

    MAKE_VECTOR_TYPE(UNIT_TYPE, 4) input_val = (MAKE_VECTOR_TYPE(UNIT_TYPE, 4))(UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO, UNIT_VAL_ZERO);

    const int2 coord = (int2)(iyx, o);
    uint input_idx = o * INPUT0_OFM_PITCH + iyx*4;
    
    input_val.s0 = TO_OUTPUT_TYPE(input[input_idx]);
    if(iyx*4 + 1 < INPUT0_OFM_PITCH)
        input_val.s1 = TO_OUTPUT_TYPE(input[input_idx+1]);
    if(iyx*4 + 2 < INPUT0_OFM_PITCH)
        input_val.s2 = TO_OUTPUT_TYPE(input[input_idx+2]);
    if(iyx*4 + 3 < INPUT0_OFM_PITCH)
        input_val.s3 = TO_OUTPUT_TYPE(input[input_idx+3]);
    IMAGE_WRITE(output, coord, input_val);
}
