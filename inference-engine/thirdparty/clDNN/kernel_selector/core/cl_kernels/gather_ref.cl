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

KERNEL(gather_ref)(const __global UNIT_TYPE* dictionary, const __global float* indices, __global UNIT_TYPE* output)
{
    const uint workItemId = get_global_id(0);

    if (workItemId >= COMPUTATIONAL_OPERATIONS_NUMBER)
        return;

    uint partNumber = workItemId / INPUT1_LENGTH;
    uint outputIndex = workItemId * SLICE_SIZE;
    uint index = workItemId - (partNumber * INPUT1_LENGTH);

    for (int k = 0; k < SLICE_SIZE; ++k)
    {
        output[outputIndex++] = dictionary[(partNumber * PART_SIZE) + ((uint) indices[index] * SLICE_SIZE) + k];
    }
}
