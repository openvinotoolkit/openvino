// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL (mvn_gpu_ref_within_channels)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    float mean = 0;

    const uint input_first = INPUT0_OFFSET + b * INPUT0_BATCH_PITCH + f * INPUT0_FEATURE_PITCH;

    // Compute mean
    uint input_idx = input_first;
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
#elif INPUT0_SIMPLE
                mean += (float)input[input_idx];
                input_idx += INPUT0_X_PITCH;
            }
            input_idx += INPUT0_Y_PITCH - INPUT0_SIZE_X*INPUT0_X_PITCH;
        }
        input_idx += INPUT0_Z_PITCH - INPUT0_SIZE_Y*INPUT0_Y_PITCH;
#endif
    }
    mean /= INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z;

    uint output_idx = OUTPUT_OFFSET + b * OUTPUT_BATCH_PITCH + f * OUTPUT_FEATURE_PITCH;

#if NORMALIZE_VARIANCE == 0
    //subtract mean
    input_idx = input_first;
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
#elif INPUT0_SIMPLE
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
#if defined EPS_OUTSIDE_SQRT
    variance = native_powr(native_sqrt(variance) + (float)EPSILON, -1.f);
#elif defined EPS_INSIDE_SQRT
    variance = native_powr(variance + (float)EPSILON, -0.5f);
#endif

    input_idx = input_first;
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
        input_idx += INPUT0_Z_PITCH - INPUT0_SIZE_Y*INPUT0_Y_PITCH;
        output_idx += OUTPUT_Z_PITCH - INPUT0_SIZE_Y*OUTPUT_Y_PITCH;
#endif
    }
#endif //NORMALIZE_VARIANCE
}
