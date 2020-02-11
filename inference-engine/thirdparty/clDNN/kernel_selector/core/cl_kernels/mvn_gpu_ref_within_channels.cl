// Copyright (c) 2018-2019 Intel Corporation
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

#include "include/fetch.cl"
#include "include/data_types.cl"


#if OUTPUT_IS_FP && INPUT0_IS_FP
    #if FP16_UNIT_USED
        #define UNIT_CVT_FUNC(val) convert_half(val)
    #else
        #define UNIT_CVT_FUNC(val) (val)
    #endif
#else
    #define UNIT_CVT_FUNC(val) convert_float(val)
#endif


KERNEL (mvn_gpu_ref_within_channels)(const __global INPUT0_TYPE* input, __global OUTPUT_TYPE* output)
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    float mean = 0.f;

    const uint input_first = INPUT0_OFFSET + b * INPUT0_BATCH_PITCH + f * INPUT0_FEATURE_PITCH;

    // Compute mean
    uint input_idx = input_first;
    for (uint z = 0; z < INPUT0_SIZE_Z; z++)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
#if INPUT0_LAYOUT_BFZYX_F16 || INPUT0_LAYOUT_BFZYX_B16F16
                input_idx = INPUT0_GET_INDEX( b, f, z, y, x);
                mean += (float)input[input_idx];
             }
        }
#else
                mean += (float)input[input_idx];
                input_idx += INPUT0_X_PITCH;
            }
            input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
        }
        input_idx += INPUT0_Z_PITCH - INPUT0_SIZE_Y*INPUT0_Y_PITCH;
#endif
    }
    mean /= INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z;

#if INPUT0_LAYOUT_BFZYX_F16 || INPUT0_LAYOUT_BFZYX_B16F16
    uint output_idx;
#else
    uint output_idx = OUTPUT_OFFSET + b * OUTPUT_BATCH_PITCH + f * OUTPUT_FEATURE_PITCH;
#endif
#if NORMALIZE_VARIANCE == 0
    //subtract mean
    input_idx = input_first;
    for (uint z = 0; z < INPUT0_SIZE_Z; z++)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
#if INPUT0_LAYOUT_BFZYX_F16 || INPUT0_LAYOUT_BFZYX_B16F16
                input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
                output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
                output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(UNIT_CVT_FUNC(input[input_idx]) - UNIT_CVT_FUNC(mean), ACTIVATION_PARAMS));
            }
        }
#else
                output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(UNIT_CVT_FUNC(input[input_idx]) - UNIT_CVT_FUNC(mean), ACTIVATION_PARAMS));
                input_idx += INPUT0_X_PITCH;
                output_idx += OUTPUT_X_PITCH;
            }
            input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
            output_idx += OUTPUT_Y_PITCH - INPUT0_SIZE_X*OUTPUT_X_PITCH;
        }
        input_idx += INPUT0_Z_PITCH - INPUT0_SIZE_Y*INPUT0_Y_PITCH;
        output_idx += OUTPUT_Z_PITCH - INPUT0_SIZE_Y*OUTPUT_Y_PITCH;
#endif
    }
#else //NORMALIZE_VARIANCE
    float variance = 0.f;

    //compute variance
    input_idx = input_first;
    for (uint z = 0; z < INPUT0_SIZE_Z; z++)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
#if INPUT0_LAYOUT_BFZYX_F16 || INPUT0_LAYOUT_BFZYX_B16F16
                input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
                float res = (float)input[input_idx] - mean;
                variance = fma(res, res, variance);
            }
        }
#else
                float res = (float)input[input_idx] - mean;
                variance = fma(res, res, variance);
                input_idx += INPUT0_X_PITCH;
            }
            input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
        }
        input_idx += INPUT0_Z_PITCH - INPUT0_SIZE_Y*INPUT0_Y_PITCH;
#endif
    }

    //normalize variance
    variance /= INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X;
    variance = native_powr(variance + (float)EPSILON, -0.5f);

    input_idx = input_first;
    for (uint z = 0; z < INPUT0_SIZE_Z; z++)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; y++)
        {
            for (uint x = 0; x < INPUT0_SIZE_X; x++)
            {
#if INPUT0_LAYOUT_BFZYX_F16 || INPUT0_LAYOUT_BFZYX_B16F16
                input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
                output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
                output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION((UNIT_CVT_FUNC(input[input_idx]) - UNIT_CVT_FUNC(mean)) * UNIT_CVT_FUNC(variance), ACTIVATION_PARAMS));
            }
        }
#else
                output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION((UNIT_CVT_FUNC(input[input_idx]) - UNIT_CVT_FUNC(mean)) * UNIT_CVT_FUNC(variance), ACTIVATION_PARAMS));
                input_idx += INPUT0_X_PITCH;
                output_idx += OUTPUT_X_PITCH;
            }
            input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
            output_idx += OUTPUT_Y_PITCH - INPUT0_SIZE_X*OUTPUT_X_PITCH;
        }
        input_idx += INPUT0_Z_PITCH - INPUT0_SIZE_Y*INPUT0_Y_PITCH;
        output_idx += OUTPUT_Z_PITCH - INPUT0_SIZE_Y*OUTPUT_Y_PITCH;
#endif
    }
#endif
}


#undef UNIT_CVT_FUNC
