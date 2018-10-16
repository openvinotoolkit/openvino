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


#include "include/include_all.cl"

KERNEL(lookup_table)(const __global UNIT_TYPE* input0, const __global float* indices, __global UNIT_TYPE* output)
{
    const uint x    = (uint)get_global_id(0);
    const uint b    = (uint)get_global_id(1);
	const uint size = INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_FEATURE_NUM;
    #ifdef INPUT0_LAYOUT_BFYX
    const uint global_index = b * VAL_NUM + x;
    output[global_index] = input0[(int)indices[global_index] + b*size];
    #elif defined INPUT0_LAYOUT_YXFB
    const uint global_index = b + x * INPUT0_BATCH_NUM;
    output[global_index] = input0[(int)indices[global_index]*INPUT0_BATCH_NUM + b];
    #endif
}
	