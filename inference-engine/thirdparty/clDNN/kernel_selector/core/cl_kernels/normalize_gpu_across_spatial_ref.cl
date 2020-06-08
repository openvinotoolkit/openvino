// Copyright (c) 2016-2020 Intel Corporation
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

KERNEL (normalize_gpu_across_spatial_bfyx)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
#if HAS_FUSED_OPS_DECLS
    FUSED_OPS_DECLS,
#endif
    const __global SCALE_TABLE_TYPE* scale_input
    )
{
    const uint b = get_global_id(0);

    float norm = EPSILON;

    const uint input_first = INPUT0_OFFSET + b * INPUT0_BATCH_PITCH;

    // Compute norm
    uint input_idx = input_first;
    for (uint f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
                float value = (float)input[input_idx];
                norm = mad(value, value, norm);
                input_idx += INPUT0_X_PITCH;
            }
            input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
        }
        input_idx += INPUT0_FEATURE_PITCH - INPUT0_Y_PITCH*INPUT0_SIZE_Y;
    }
    if(norm <= THRESHOLD)
    {
        norm = 0;
    }
    else
    {
        norm = native_powr(norm, -0.5f);
    }

    uint output_idx = OUTPUT_OFFSET + b * OUTPUT_BATCH_PITCH;

    // Scale the input
    input_idx = input_first;
    for (uint f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
#if SCALE_TABLE_FEATURE_NUM == 1
        const uint scale_index = 0;
#elif INPUT0_FEATURE_NUM <= SCALE_TABLE_FEATURE_NUM
        const uint scale_index = f;
#else
        const uint scale_index = f % SCALE_TABLE_FEATURE_NUM;
#endif

        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
                ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(norm) * TO_ACTIVATION_TYPE(input[input_idx]) * TO_ACTIVATION_TYPE(scale_input[scale_index]);
#if HAS_FUSED_OPS
                FUSED_OPS;
                output[output_idx] = FUSED_OPS_RESULT;
#else
                output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#endif
                input_idx += INPUT0_X_PITCH;
                output_idx += OUTPUT_X_PITCH;
            }
            input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
            output_idx += OUTPUT_Y_PITCH - INPUT0_SIZE_X*OUTPUT_X_PITCH;
        }
        input_idx += INPUT0_FEATURE_PITCH - INPUT0_Y_PITCH*INPUT0_SIZE_Y;
        output_idx += OUTPUT_FEATURE_PITCH - INPUT0_SIZE_Y*OUTPUT_Y_PITCH;
    }
}
