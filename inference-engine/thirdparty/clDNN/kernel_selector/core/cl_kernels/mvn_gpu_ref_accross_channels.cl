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


#if FP16_UNIT_USED
    #define UNIT_CVT_FUNC(val) convert_half(val)
#else
    #define UNIT_CVT_FUNC(val) (val)
#endif


KERNEL (mvn_gpu_ref_accross_channels)(const __global UNIT_TYPE* input, __global UNIT_TYPE* output)
{
    const uint b = get_global_id(0);
    float mean = 0.f;

    const uint input_first = INPUT0_OFFSET + b * INPUT0_BATCH_PITCH;

    // Compute mean
    uint input_idx = input_first;
    for (uint f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        for (uint z = 0; z < INPUT0_SIZE_Z; z++)
        {
            for (uint y = 0; y < INPUT0_SIZE_Y; y++)
            {
                for (uint x = 0; x < INPUT0_SIZE_X; x++)
                {
#if INPUT0_LAYOUT_BFZYX_F16
                    input_idx = GET_DATA_BFZYX_F16_INDEX(INPUT0, b, f, z, y, x);
                    mean += (float)input[input_idx];
                }
            }
        }
#else
                    mean += (float)input[input_idx];
                    input_idx += INPUT0_X_PITCH;
                }
                input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
            }
            input_idx += INPUT0_Z_PITCH - INPUT0_Y_PITCH*INPUT0_SIZE_Y;
        }
        input_idx += INPUT0_FEATURE_PITCH - INPUT0_Z_PITCH*INPUT0_SIZE_Z;
#endif
    }
#if INPUT0_LAYOUT_BFZYX_F16
    uint output_idx;
#else
    uint output_idx = OUTPUT_OFFSET + b * OUTPUT_BATCH_PITCH;
#endif
    mean /= INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X;

#if NORMALIZE_VARIANCE == 0
    // Subtract mean / compute variance if needed
    input_idx = input_first;
    for (uint f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        for (uint z = 0; z < INPUT0_SIZE_Z; z++)
        {
            for (uint y = 0; y < INPUT0_SIZE_Y; y++)
            {
                for (uint x = 0; x < INPUT0_SIZE_X; x++)
                {
#if INPUT0_LAYOUT_BFZYX_F16
                    input_idx = GET_DATA_BFZYX_F16_INDEX(INPUT0, b, f, z, y, x);
                    output_idx = GET_DATA_BFZYX_F16_INDEX(OUTPUT, b, f, z, y, x);
                    output[output_idx] = ACTIVATION(input[input_idx] - UNIT_CVT_FUNC(mean), ACTIVATION_PARAMS);
                }
            }
        }
#else
                    output[output_idx] = ACTIVATION(input[input_idx] - UNIT_CVT_FUNC(mean), ACTIVATION_PARAMS);
                    input_idx += INPUT0_X_PITCH;
                    output_idx += OUTPUT_X_PITCH;
                }
                input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
                output_idx += OUTPUT_Y_PITCH - INPUT0_SIZE_X*OUTPUT_X_PITCH;
            }
            input_idx += INPUT0_Z_PITCH - INPUT0_Y_PITCH*INPUT0_SIZE_Y;
            output_idx += OUTPUT_Z_PITCH - INPUT0_SIZE_Y*OUTPUT_Y_PITCH;
        }
        input_idx += INPUT0_FEATURE_PITCH - INPUT0_Z_PITCH*INPUT0_SIZE_Z;
        output_idx += OUTPUT_FEATURE_PITCH - INPUT0_SIZE_Z*OUTPUT_Z_PITCH;
#endif
    }

#else //NORMALIZE_VARIANCE
    float variance = 0.f;

    //compute variance
    input_idx = input_first;
    for (uint f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        for (uint z = 0; z < INPUT0_SIZE_Z; z++)
        {
            for (uint y = 0; y < INPUT0_SIZE_Y; y++)
            {
                for (uint x = 0; x < INPUT0_SIZE_X; x++)
                {
#if INPUT0_LAYOUT_BFZYX_F16
                    input_idx = GET_DATA_BFZYX_F16_INDEX(INPUT0, b, f, z, y, x);
                    float res = (float)input[input_idx] - mean;
                    variance = fma(res, res, variance);
                }
            }
        }
#else
                    float res = (float)input[input_idx] - mean;
                    variance = fma(res, res, variance);
                    input_idx += INPUT0_X_PITCH;
                }
                input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
            }
            input_idx += INPUT0_Z_PITCH - INPUT0_Y_PITCH*INPUT0_SIZE_Y;
        }
        input_idx += INPUT0_FEATURE_PITCH - INPUT0_Z_PITCH*INPUT0_SIZE_Z;
#endif
    }

    //normalize variance
    variance /= INPUT0_FEATURE_NUM * INPUT0_SIZE_Z * INPUT0_SIZE_Y * INPUT0_SIZE_X;
    variance = native_powr(variance + (float)EPSILON, -0.5f);

    input_idx = input_first;
    for (uint f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        for (uint z = 0; z < INPUT0_SIZE_Z; z++)
        {
            for (uint y = 0; y < INPUT0_SIZE_Y; y++)
            {
                for (uint x = 0; x < INPUT0_SIZE_X; x++)
                {
#if INPUT0_LAYOUT_BFZYX_F16
                    input_idx = GET_DATA_BFZYX_F16_INDEX(INPUT0, b, f, z, y, x);
                    output_idx = GET_DATA_BFZYX_F16_INDEX(OUTPUT, b, f, z, y, x);
                    output[output_idx] = ACTIVATION((input[input_idx] - UNIT_CVT_FUNC(mean)) * UNIT_CVT_FUNC(variance), ACTIVATION_PARAMS);
                }
            }
        }
#else
                    output[output_idx] = ACTIVATION((input[input_idx] - UNIT_CVT_FUNC(mean)) * UNIT_CVT_FUNC(variance), ACTIVATION_PARAMS);
                    input_idx += INPUT0_X_PITCH;
                    output_idx += OUTPUT_X_PITCH;
                }
                input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
                output_idx += OUTPUT_Y_PITCH - INPUT0_SIZE_X*OUTPUT_X_PITCH;
            }
            input_idx += INPUT0_Z_PITCH - INPUT0_Y_PITCH*INPUT0_SIZE_Y;
            output_idx += OUTPUT_Z_PITCH - INPUT0_SIZE_Y*OUTPUT_Y_PITCH;
        }
        input_idx += INPUT0_FEATURE_PITCH - INPUT0_Z_PITCH*INPUT0_SIZE_Z;
        output_idx += OUTPUT_FEATURE_PITCH - INPUT0_SIZE_Z*OUTPUT_Z_PITCH;
#endif
    }
#endif
}


#undef UNIT_CVT_FUNC
