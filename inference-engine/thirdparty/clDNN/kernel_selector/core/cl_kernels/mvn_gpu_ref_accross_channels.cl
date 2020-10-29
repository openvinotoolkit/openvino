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


KERNEL (mvn_gpu_ref_accross_channels)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint b = get_global_id(0);
    float mean = 0;

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
#if !INPUT0_SIMPLE
#   if INPUT0_DIMS <= 4
                    input_idx = INPUT0_GET_INDEX(b, f, y, x);
#   elif INPUT0_DIMS == 5
                    input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
#   endif
                    mean += (float)input[input_idx];
                }
            }
        }
#elif INPUT0_SIMPLE
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

    uint output_idx = OUTPUT_OFFSET + b * OUTPUT_BATCH_PITCH;
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
#if !INPUT0_SIMPLE || !OUTPUT_SIMPLE
#   if INPUT0_DIMS <= 4
                    input_idx = INPUT0_GET_INDEX(b, f, y, x);
                    output_idx = OUTPUT_GET_INDEX(b, f, y, x);
#   elif INPUT0_DIMS == 5
                    input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
                    output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
#   endif

                    ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(input[input_idx]) - TO_ACTIVATION_TYPE(mean);
#   if HAS_FUSED_OPS
                    FUSED_OPS;
                    output[output_idx] = FUSED_OPS_RESULT;
#   else
                    output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
                }
            }
        }
#elif INPUT0_SIMPLE && OUTPUT_SIMPLE
                    ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(input[input_idx]) - TO_ACTIVATION_TYPE(mean);
#   if HAS_FUSED_OPS
                    FUSED_OPS;
                    output[output_idx] = FUSED_OPS_RESULT;
#   else
                    output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
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
#if !INPUT0_SIMPLE
#   if INPUT0_DIMS <= 4
                    input_idx = INPUT0_GET_INDEX(b, f, y, x);
#   elif INPUT0_DIMS == 5
                    input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
#   endif
                    float res = (float)input[input_idx] - mean;
                    variance = fma(res, res, variance);
                }
            }
        }
#elif INPUT0_SIMPLE
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
#if !INPUT0_SIMPLE || !OUTPUT_SIMPLE
#   if INPUT0_DIMS <= 4
                    input_idx = INPUT0_GET_INDEX(b, f, y, x);
                    output_idx = OUTPUT_GET_INDEX(b, f, y, x);
#   elif INPUT0_DIMS == 5
                    input_idx = INPUT0_GET_INDEX(b, f, z, y, x);
                    output_idx = OUTPUT_GET_INDEX(b, f, z, y, x);
#   endif

                    ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(input[input_idx]) - TO_ACTIVATION_TYPE(mean)) * TO_ACTIVATION_TYPE(variance);
#   if HAS_FUSED_OPS
                    FUSED_OPS;
                    output[output_idx] = FUSED_OPS_RESULT;
#   else
                    output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
                }
            }
        }
#elif INPUT0_SIMPLE && OUTPUT_SIMPLE
                    ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(input[input_idx]) - TO_ACTIVATION_TYPE(mean)) * TO_ACTIVATION_TYPE(variance);
#   if HAS_FUSED_OPS
                    FUSED_OPS;
                    output[output_idx] = FUSED_OPS_RESULT;
#   else
                    output[output_idx] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
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
