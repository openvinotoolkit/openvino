// Copyright (c) 2019-2020 Intel Corporation
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

KERNEL(shuffle_channels_ref)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
    uint dimensions[] = { batch, feature, y, x };

    const uint current_group = dimensions[AXIS] / GROUP_SIZE;
    const uint position_in_group = dimensions[AXIS] % GROUP_SIZE;
    const uint input_index = INPUT0_GET_INDEX(batch, feature, y, x);

    dimensions[AXIS] = (position_in_group * GROUPS_NUMBER + current_group);
    uint output_index = OUTPUT_GET_INDEX(dimensions[0], dimensions[1], dimensions[2], dimensions[3]);

    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}
