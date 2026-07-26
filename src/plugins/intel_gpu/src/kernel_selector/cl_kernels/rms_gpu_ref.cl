// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

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
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    const uint w = 0;

    // INPUT_RANK selects the logical normalization axis.
#if INPUT_RANK >= 4
    const uint outer_z_size = INPUT0_SIZE_Z;
    const uint outer_y_size = INPUT0_SIZE_Y;
    const uint norm_size = INPUT0_SIZE_X;
#else
    const uint outer_z_size = 1;
    const uint outer_y_size = 1;
    const uint norm_size = INPUT0_SIZE_X * INPUT0_SIZE_Y * INPUT0_SIZE_Z;
#endif

    for (uint outer_z = 0; outer_z < outer_z_size; outer_z++) {
        for (uint outer_y = 0; outer_y < outer_y_size; outer_y++) {
#if INPUT_RANK >= 4
            const uint z_begin = outer_z;
            const uint z_end = outer_z + 1;
            const uint y_begin = outer_y;
            const uint y_end = outer_y + 1;
#else
            const uint z_begin = 0;
            const uint z_end = INPUT0_SIZE_Z;
            const uint y_begin = 0;
            const uint y_end = INPUT0_SIZE_Y;
#endif
            ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;
            for (uint z = z_begin; z < z_end; z++) {
                for (uint y = y_begin; y < y_end; y++) {
                    for (uint x = 0; x < INPUT0_SIZE_X; x++) {
                        const uint input_idx = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
                        rms += pow(TO_ACCUMULATOR_TYPE(input[input_idx]), 2);
                    }
                }
            }

            rms /= norm_size;
            rms = pow(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);

            for (uint z = z_begin; z < z_end; z++) {
                for (uint y = y_begin; y < y_end; y++) {
                    for (uint x = 0; x < INPUT0_SIZE_X; x++) {
                        const uint input_idx = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
                        const uint output_idx = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
#if ELEMENTWISE_AFFINE
                        const uint gamma_idx = INPUT1_LENGTH == 1 ? 0 : GAMMA_AXIS_INDEX;
                        OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms) * TO_OUTPUT_TYPE(input[input_idx]) * TO_OUTPUT_TYPE(gamma[gamma_idx]);
#else
                        OUTPUT_TYPE result = TO_OUTPUT_TYPE(rms) * TO_OUTPUT_TYPE(input[input_idx]);
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
    }
}
