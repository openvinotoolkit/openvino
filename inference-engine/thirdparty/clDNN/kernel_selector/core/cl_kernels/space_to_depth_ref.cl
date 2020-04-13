// Copyright (c) 2020 Intel Corporation
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

KERNEL(space_to_depth_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;

    const uint input_offset = BLOCKS_FIRST_MODE * (feature / INPUT0_FEATURE_NUM) + (!BLOCKS_FIRST_MODE) * (feature % SQUARED_BLOCK_SIZE);
    const uint input_y = (y * BLOCK_SIZE) + (input_offset / BLOCK_SIZE);
    const uint input_x = (x * BLOCK_SIZE) + (input_offset % BLOCK_SIZE);

    const uint input_feature = BLOCKS_FIRST_MODE * (feature % INPUT0_FEATURE_NUM) + (!BLOCKS_FIRST_MODE) * (feature / SQUARED_BLOCK_SIZE);
    const uint input_feature_offset = (input_y * INPUT0_Y_PITCH) + input_x;

    const uint input_index = INPUT0_OFFSET + (batch * INPUT0_BATCH_PITCH) + (input_feature * INPUT0_FEATURE_PITCH) + input_feature_offset;
    const uint output_index = OUTPUT_OFFSET + (batch * OUTPUT_BATCH_PITCH) + (feature * OUTPUT_FEATURE_PITCH) + (y * OUTPUT_Y_PITCH) + x;

    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}
