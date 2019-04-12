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

#define GET_COORDS_INDEX(prefix, coords) GET_DATA_INDEX(prefix, coords[0], coords[1], coords[2], coords[3])

KERNEL(one_hot_ref)(
    const __global INPUT0_TYPE* input,
    __global INPUT0_TYPE* output)
{
    uint in_coords[4] = { 0, get_global_id(0), get_global_id(1), get_global_id(2) };
    uint out_coords[4] = { 0, get_global_id(0), get_global_id(1), get_global_id(2) };
    for (uint i = 0; i < ONE_HOT_AXIS; ++i)
        out_coords[i] = out_coords[i + 1];

    // Fill the output with 0
    for (out_coords[ONE_HOT_AXIS] = 0; out_coords[ONE_HOT_AXIS] < ONE_HOT_LIMIT; ++out_coords[ONE_HOT_AXIS])
        output[GET_COORDS_INDEX(OUTPUT, out_coords)] = 0;

    // Put in the 1; ignore bad input values
    INPUT0_TYPE val = input[GET_COORDS_INDEX(INPUT0, in_coords)];
    if (val >= 0 && val < ONE_HOT_LIMIT)
    {
        out_coords[ONE_HOT_AXIS] = val;
        output[GET_COORDS_INDEX(OUTPUT, out_coords)] = 1;
    }
}
