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

#if OUTPUT_DIMS == 5
#define SPATIAL_BLOCK_SIZE (BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE)
#else
#define SPATIAL_BLOCK_SIZE (BLOCK_SIZE*BLOCK_SIZE)
#endif

KERNEL(space_to_depth_ref)(const __global INPUT0_TYPE* input,
                                 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);

#if OUTPUT_DIMS == 5
    const uint z = ((uint)get_global_id(2) / OUTPUT_SIZE_X) / OUTPUT_SIZE_Y;
    const uint y = ((uint)get_global_id(2) / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#else
    const uint z = 0;
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
#endif

#if BLOCKS_FIRST_MODE
    const uint input_offset = feature / INPUT0_FEATURE_NUM;
    const uint input_feature = feature % INPUT0_FEATURE_NUM;
#else
    const uint input_offset = feature % SPATIAL_BLOCK_SIZE;
    const uint input_feature = feature / SPATIAL_BLOCK_SIZE;
#endif

#if OUTPUT_DIMS == 5
    const uint input_z = (z * BLOCK_SIZE) + ((input_offset / BLOCK_SIZE) / BLOCK_SIZE);
    const uint input_y = (y * BLOCK_SIZE) + ((input_offset / BLOCK_SIZE) % BLOCK_SIZE);
    const uint input_x = (x * BLOCK_SIZE) + (input_offset % BLOCK_SIZE);
    const uint input_index = INPUT0_GET_INDEX(batch, input_feature, input_z, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, z, y, x);
#else
    const uint input_z = 0;
    const uint input_y = (y * BLOCK_SIZE) + (input_offset / BLOCK_SIZE);
    const uint input_x = (x * BLOCK_SIZE) + (input_offset % BLOCK_SIZE);
    const uint input_index = INPUT0_GET_INDEX(batch, input_feature, input_y, input_x);
    const uint output_index = OUTPUT_GET_INDEX(batch, feature, y, x);
#endif

    INPUT0_TYPE in_val = input[input_index];
#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_index] = FUSED_OPS_RESULT;
#else
    output[output_index] = ACTIVATION(in_val, ACTIVATION_PARAMS);
#endif
}
