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

KERNEL(shuffle_channels_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
    const uint y = (uint)get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = (uint)get_global_id(2) % OUTPUT_SIZE_X;
    const uint dimensions[] = { batch, feature, y, x };

    const uint current_group = dimensions[AXIS] / GROUP_SIZE;
    const uint position_in_group = dimensions[AXIS] % GROUP_SIZE;
    const uint input_index = INPUT0_OFFSET + (batch * INPUT0_BATCH_PITCH) + (feature * INPUT0_FEATURE_PITCH) + (y * INPUT0_Y_PITCH) + x;

    uint output_index = OUTPUT_OFFSET;

    for (uint i = 0; i < AXIS; ++i) {
        output_index += dimensions[i] * INPUT0_PITCHES[INPUT0_DIMS - i - 1];
    }

    output_index += (position_in_group * GROUPS_NUMBER + current_group) * INPUT0_PITCHES[INPUT0_DIMS - AXIS - 1];

    for (uint i = AXIS + 1; i < INPUT0_DIMS; ++i) {
        output_index += dimensions[i] * INPUT0_PITCHES[INPUT0_DIMS - i - 1];
    }

    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
}
