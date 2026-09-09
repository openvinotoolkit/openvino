// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(activation)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const unsigned int x = (uint)get_global_id(0) * NUM_COLS_WI;
    unsigned int input_offset  = x + INPUT0_OFFSET;
    unsigned int output_offset = x + OUTPUT_OFFSET;

    typedef CAT(INPUT0_TYPE, 4) input_t;
	typedef CAT(INPUT0_COMPUTE_TYPE, 4) input_compute_t;
    typedef CAT(OUTPUT_TYPE, 4) output_t;

    input_t v = ((__global input_t*) (input + input_offset))[0];

    input_compute_t v_compute = ACTIVATION_KERNEL(DECODE_INPUT0_COMPUTE_VECTOR_TYPE(v, 4), ACTIVATION_PARAMS_KERNEL);

#if HAS_FUSED_OPS
    output_t result;
	v = TO_INPUT0_VECTOR_TYPE(v_compute, 4);
    #if !CAN_USE_VECTOR
        for (int i = 0; i < 4; i++) {
            FUSED_OPS_SCALAR;
            result[i] = FUSED_OPS_RESULT_SCALAR;
        }
    #else
        FUSED_OPS_VECTOR;
        result = FUSED_OPS_RESULT_VECTOR;
    #endif
    *((__global output_t*)(output + output_offset)) = result;
#else
    *((__global output_t*)(output + output_offset)) = TO_OUTPUT_VECTOR_TYPE(v_compute, 4);
#endif
}
