// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"
#include "include/batch_headers/bf16_utils.cl"

#if NORMALIZE_BATCH
    #define NORM_SIZE INPUT0_BATCH_NUM
    #define NORM_INDEX b
#elif NORMALIZE_FEATURE
    #define NORM_SIZE INPUT0_FEATURE_NUM
    #define NORM_INDEX f
#elif NORMALIZE_Y
    #define NORM_SIZE INPUT0_SIZE_Y
    #define NORM_INDEX y
#else
    #define NORM_SIZE INPUT0_SIZE_X
    #define NORM_INDEX x
#endif

KERNEL(rms_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if ELEMENTWISE_AFFINE
    const __global INPUT1_TYPE* gamma,
#endif
    __global OUTPUT_TYPE* output
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_DECLS
    #endif
)
{
#if NORMALIZE_X
    const uint outer_z_size = INPUT0_SIZE_Z;
    const uint outer_y_size = INPUT0_SIZE_Y;
#else
    const uint outer_z_size = 1;
    const uint outer_y_size = 1;
#endif

    for (uint outer_z = 0; outer_z < outer_z_size; outer_z++) {
        for (uint outer_y = 0; outer_y < outer_y_size; outer_y++) {
            uint b = get_global_id(0);
            uint f = get_global_id(1);
            uint z = outer_z;
            uint y = outer_y;
            uint x = 0;

            ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;
            for (uint n = 0; n < NORM_SIZE; n++) {
                NORM_INDEX = n;
                const uint input_idx = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, z, y, x);
                const ACCUMULATOR_TYPE value = TO_ACCUMULATOR_TYPE(input[input_idx]);
                rms += value * value;
            }

            rms /= NORM_SIZE;
            rms = pow(sqrt(rms + EPSILON), -1);

            for (uint n = 0; n < NORM_SIZE; n++) {
                NORM_INDEX = n;
                const uint input_idx = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, z, y, x);
                const uint output_idx = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, 0, z, y, x);
#if ELEMENTWISE_AFFINE
                const uint gamma_idx = INPUT1_OFFSET + (INPUT1_LENGTH == 1 ? 0 : n);
                OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms * TO_ACCUMULATOR_TYPE(input[input_idx]) * TO_ACCUMULATOR_TYPE(gamma[gamma_idx]));
#else
                OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms * TO_ACCUMULATOR_TYPE(input[input_idx]));
#endif
                #if HAS_FUSED_OPS
                    FUSED_OPS;
                    result = FUSED_OPS_RESULT;
                #endif
                output[output_idx] = result;
            }
        }
    }
}

#undef NORM_SIZE
#undef NORM_INDEX
