// Copyright (c) 2016-2019 Intel Corporation
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


#include "include/fetch.cl"
#include "include/unit_type.cl"

#define WORK_GROUP_SIZE 16
#define IC_BLOCK 16

__attribute__((reqd_work_group_size(1, WORK_GROUP_SIZE, 1)))
__attribute__((intel_reqd_sub_group_size(WORK_GROUP_SIZE)))
KERNEL (concatenation_gpu_blocked)(__global UNIT_TYPE* input, __global UNIT_TYPE* output, uint output_offset_in_concat_axis)
{
    const int b = get_global_id(0);
    const int f_block = get_group_id(1);
    const int xy = get_global_id(2);
    const int lid = get_sub_group_local_id();

    const int x = xy % OUTPUT_SIZE_X;
    const int y = xy / OUTPUT_SIZE_X;


#if ALIGNED
    const uint input_offset = INPUT0_GET_INDEX(b, f_block*IC_BLOCK, y, x);
    const uint dst_index = OUTPUT_GET_INDEX(b, (f_block*IC_BLOCK + output_offset_in_concat_axis), y, x);

    UNIT_TYPE src = UNIT_BLOCK_READ(input, input_offset);
    src = ACTIVATION(src, ACTIVATION_PARAMS);
    UNIT_BLOCK_WRITE(output, dst_index, src);
#else
    if (f_block*IC_BLOCK + lid >= INPUT0_FEATURE_NUM)
        return;

    const uint input_offset = INPUT0_GET_INDEX(b, f_block*IC_BLOCK + lid, y, x);
    const uint dst_index = OUTPUT_GET_INDEX(b, (f_block*IC_BLOCK + lid + output_offset_in_concat_axis), y, x);

    UNIT_TYPE src = input[input_offset];
    src = ACTIVATION(src, ACTIVATION_PARAMS);
    output[dst_index] = src;
#endif
}

#undef WORK_GROUP_SIZE
#undef IC_BLOCK
