// Copyright (c) 2018-2020 Intel Corporation
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
#include "include/fetch.cl"

KERNEL(tile_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
    const uint y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
    const uint f = (uint)get_global_id(2) / OUTPUT_BATCH_NUM;
    const uint b = (uint)get_global_id(2) % OUTPUT_BATCH_NUM;
    #if OUTPUT_DIMS == 6
    const uint z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
    const uint w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
    const uint out_offset = OUTPUT_GET_INDEX(b, f, w, z, y, x);
    const uint in_offset = INPUT0_GET_INDEX_SAFE(b, f, w, z, y, x);
    #elif OUTPUT_DIMS == 5
    const uint z = (uint)get_global_id(1);
    const uint out_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
    const uint in_offset = INPUT0_GET_INDEX_SAFE(b, f, z, y, x);
    #elif OUTPUT_DIMS == 4
    const uint out_offset = OUTPUT_GET_INDEX(b, f, y, x);
    const uint in_offset = INPUT0_GET_INDEX_SAFE(b, f, y, x);
    #endif

    output[out_offset] = input[in_offset];
}
