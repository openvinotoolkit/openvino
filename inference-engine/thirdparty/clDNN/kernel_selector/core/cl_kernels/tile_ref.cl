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

#include "include/common.cl"
#include "include/data_types.cl"

#if DENSE
__attribute__((intel_reqd_sub_group_size(16)))
__attribute__((reqd_work_group_size(16, 1, 1)))
#endif
KERNEL (tile_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
#if DENSE

    const uint id = get_global_id(0);
    const uint group_id = id / 16;
    const uint lid = get_local_id(0);
    const uint idx = min((uint)(id), (uint)(OUTER_SIZE - 1));
    UNIT_TYPE val = input[idx];

    for (int t = 0; t < TILES; t++)
    {
        UNIT_TYPE save_val = intel_sub_group_shuffle(val, (t*16 + lid)/TILES);
        int offset = group_id*16*TILES + t*16 + lid;
        if (offset < OUTPUT_SIZE)
            output[offset] = save_val;
    }
#else
    const uint outer_idx = get_global_id(0);
    const uint inner_idx = get_global_id(1);
    if (inner_idx >= AXIS_PITCH) return;

    for (int t = 0; t < TILES; t++)
    {
        output[outer_idx*TILES*AXIS_PITCH + t*AXIS_PITCH + inner_idx] = input[outer_idx*AXIS_PITCH + inner_idx];
    }
#endif
}
