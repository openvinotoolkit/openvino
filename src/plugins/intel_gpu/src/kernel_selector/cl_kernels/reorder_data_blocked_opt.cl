// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define VLOAD_N  CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define CONVERT_LONG_VEC           CAT(convert_long, VEC_SIZE)
#define INPUT_VEC_TYPE             CAT(INPUT_REORDER_TYPE, VEC_SIZE)
#define OUTPUT_VEC_TYPE            CAT(OUTPUT_REORDER_TYPE, VEC_SIZE)
#define TO_OUTPUT_VEC_TYPE(x)      CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_OUTPUT_VEC_TYPE_SAT(x)  CAT(CAT(convert_, OUTPUT_VEC_TYPE), _sat)(x)

#if INPUT0_IS_FP && !OUTPUT_IS_FP
    #if CONVERT_TRUNCATE
        #define TO_OUTPUT_TYPED_VECTOR(x)   TO_OUTPUT_VEC_TYPE(CONVERT_LONG_VEC(x))
    #else
        #define TO_OUTPUT_TYPED_VECTOR(x)   TO_OUTPUT_VEC_TYPE_SAT(x)
    #endif
#else
    #define TO_OUTPUT_TYPED_VECTOR(x)   TO_OUTPUT_VEC_TYPE(x)
#endif

KERNEL(reorder_blocked_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT_REORDER_TYPE* input,
    __global OUTPUT_REORDER_TYPE* output
    )
{
    const uint global_id = get_global_id(0);

    #if LEFTOVER
        if (global_id == get_global_size(0) - 1) {
            size_t opt_size = global_id * (size_t)ELEMENTS_NUM;
            size_t total_size = (size_t)OUTPUT_BATCH_NUM * (size_t)OUTPUT_BATCH_PITCH;
            if ((opt_size + ELEMENTS_NUM) != total_size) {
                unroll_for (uint i = 0; i < (total_size - opt_size) ; ++i) {
                    output[opt_size + i] = TO_OUTPUT_REORDER_TYPE(ACTIVATION_TYPED(OUTPUT_REORDER,
                                                                                input[opt_size + i],
                                                                                ACTIVATION_PARAMS_TYPED));
                }
                return;
            }
        }
    #endif

    OUTPUT_VEC_TYPE res[ARRAY_SIZE];
    unroll_for (uint i = 0; i < ARRAY_SIZE ; ++i) {
        res[i] = TO_OUTPUT_TYPED_VECTOR(ACTIVATION_TYPED(OUTPUT_REORDER,
                                                     VLOAD_N(global_id * ARRAY_SIZE + i, input),
                                                     ACTIVATION_PARAMS_TYPED));
    }

    unroll_for (uint i = 0; i < ARRAY_SIZE ; ++i) {
        VSTORE_N(res[i], global_id * ARRAY_SIZE + i, output);
    }
}
