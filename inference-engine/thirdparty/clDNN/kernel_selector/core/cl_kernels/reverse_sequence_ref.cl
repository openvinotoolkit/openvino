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

KERNEL(reverse_sequence_ref)(const __global UNIT_TYPE* input, const __global float* seq_lengths, __global UNIT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);
    const uint y = get_global_id(2) / INPUT0_SIZE_X;
    const uint x = get_global_id(2) % INPUT0_SIZE_X;
    uint dimensions[] = { batch, feature, y, x };

    const uint input_index = INPUT0_OFFSET +
                             batch * INPUT0_BATCH_PITCH +
                             feature * INPUT0_FEATURE_PITCH +
                             y * INPUT0_Y_PITCH +
                             x * INPUT0_X_PITCH;

    const uint length = seq_lengths[dimensions[BATCH_AXIS]];
    if (dimensions[SEQ_AXIS] < length)
        dimensions[SEQ_AXIS] = length - dimensions[SEQ_AXIS] - 1;

    const uint output_index = OUTPUT_OFFSET +
                              dimensions[0] * OUTPUT_BATCH_PITCH +
                              dimensions[1] * OUTPUT_FEATURE_PITCH +
                              dimensions[2] * OUTPUT_Y_PITCH +
                              dimensions[3] * OUTPUT_X_PITCH;

    output[output_index] = input[input_index];
}
