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

#if ITEM_SIZE < 4
    unroll_for (uint i = 0; i < ITEM_SIZE ; ++i) {
        OUTPUT_VEC_TYPE res = TO_OUTPUT_TYPED_VECTOR(VLOAD_N(global_id * ITEM_SIZE + i, input));

        res = ACTIVATION_TYPED(OUTPUT_REORDER, res, ACTIVATION_PARAMS_TYPED);

        VSTORE_N(res, global_id * ITEM_SIZE + i, output);
    }
#else
    OUTPUT_VEC_TYPE res[ITEM_SIZE/4][4];
    unroll_for (uint i = 0; i < (ITEM_SIZE/4) ; ++i) {
        res[i][0] = TO_OUTPUT_TYPED_VECTOR(VLOAD_N(global_id*ITEM_SIZE + i*4, input));
        res[i][1] = TO_OUTPUT_TYPED_VECTOR(VLOAD_N(global_id*ITEM_SIZE + i*4 + 1, input));
        res[i][2] = TO_OUTPUT_TYPED_VECTOR(VLOAD_N(global_id*ITEM_SIZE + i*4 + 2, input));
        res[i][3] = TO_OUTPUT_TYPED_VECTOR(VLOAD_N(global_id*ITEM_SIZE + i*4 + 3, input));

        res[i][0] = ACTIVATION_TYPED(OUTPUT_REORDER, (res[i][0]), ACTIVATION_PARAMS_TYPED);
        res[i][1] = ACTIVATION_TYPED(OUTPUT_REORDER, (res[i][1]), ACTIVATION_PARAMS_TYPED);
        res[i][2] = ACTIVATION_TYPED(OUTPUT_REORDER, (res[i][2]), ACTIVATION_PARAMS_TYPED);
        res[i][3] = ACTIVATION_TYPED(OUTPUT_REORDER, (res[i][3]), ACTIVATION_PARAMS_TYPED);

        VSTORE_N(res[i][0], global_id*ITEM_SIZE + i*4, output);
        VSTORE_N(res[i][1], global_id*ITEM_SIZE + i*4 + 1, output);
        VSTORE_N(res[i][2], global_id*ITEM_SIZE + i*4 + 2, output);
        VSTORE_N(res[i][3], global_id*ITEM_SIZE + i*4 + 3, output);
    }
#endif
}
