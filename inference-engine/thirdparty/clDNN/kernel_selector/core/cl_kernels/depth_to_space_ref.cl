// Copyright (c) 2019 Intel Corporation
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

KERNEL(depth_to_space_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
    const uint y = get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = get_global_id(2) % OUTPUT_SIZE_X;

    const uint input_y = y / BLOCK_SIZE;
    const uint offset_y = y % BLOCK_SIZE;

    const uint input_x = x / BLOCK_SIZE;
    const uint offset_x = (x % BLOCK_SIZE);
    const uint offset_feature = (offset_y * BLOCK_SIZE + offset_x) * OUTPUT_FEATURE_NUM;

    const uint output_index = OUTPUT_OFFSET + (batch * OUTPUT_BATCH_PITCH) + (feature * OUTPUT_FEATURE_PITCH) + (y * OUTPUT_Y_PITCH) + x;
    const uint input_feature = feature + offset_feature;
    const uint input_index = INPUT0_OFFSET + (batch * INPUT0_BATCH_PITCH) + (input_feature * INPUT0_FEATURE_PITCH) + (input_y * INPUT0_Y_PITCH) + input_x;
    output[output_index] = input[input_index];
}
