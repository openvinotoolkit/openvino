/*
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
*/

#include "include/include_all.cl"

KERNEL (upsampling_gpu_ref)(__global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
#if defined OUTPUT_LAYOUT_YXFB
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

    const uint dst_index = batch*OUTPUT_BATCH_PITCH + feature*OUTPUT_FEATURE_PITCH + y*OUTPUT_Y_PITCH + x*OUTPUT_X_PITCH + OUTPUT_OFFSET;

    const uint src_x = floor(x * X_RATIO);
    const uint src_y = floor(y * Y_RATIO);
    const uint src_index = batch*INPUT0_BATCH_PITCH + feature*INPUT0_FEATURE_PITCH + src_y*INPUT0_Y_PITCH + src_x*INPUT0_X_PITCH + INPUT0_OFFSET;
    output[dst_index] = input[src_index];

}
