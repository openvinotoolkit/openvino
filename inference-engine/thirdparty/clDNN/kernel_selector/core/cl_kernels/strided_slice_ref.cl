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

KERNEL(strided_slice_ref)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint batch = get_global_id(0);
    const uint feature = get_global_id(1);

#if NEW_AXIS_MODE
    // If NEW_AXIS_MODE that just copy input to output
    const uint y_input = get_global_id(2) / INPUT0_SIZE_X;
    const uint x_input = get_global_id(2) % INPUT0_SIZE_X;
    const uint input_index = INPUT0_OFFSET +
        batch * INPUT0_BATCH_PITCH +
        feature * INPUT0_FEATURE_PITCH +
        y_input * INPUT0_Y_PITCH +
        x_input * INPUT0_X_PITCH;
    output[input_index] = input[input_index];
#else
    const uint y = get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = get_global_id(2) % OUTPUT_SIZE_X;
    const uint input_index = INPUT0_OFFSET +
            (SLICE_BEGIN_BATCH + batch * SLICE_STEPS_BATCH) * INPUT0_BATCH_PITCH +
            (SLICE_BEGIN_FEATURE + feature * SLICE_STEPS_FEATURE) * INPUT0_FEATURE_PITCH +
            (SLICE_BEGIN_Y + y * SLICE_STEPS_Y) * INPUT0_Y_PITCH +
            (SLICE_BEGIN_X + x * SLICE_STEPS_X) * INPUT0_X_PITCH;

    const uint output_index = OUTPUT_OFFSET +
            batch * OUTPUT_BATCH_PITCH +
            feature * OUTPUT_FEATURE_PITCH +
            y * OUTPUT_Y_PITCH +
            x * OUTPUT_X_PITCH;

    output[output_index] = ACTIVATION(input[input_index], ACTIVATION_PARAMS);
#endif
}
