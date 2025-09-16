// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define VLOAD_N CAT(vload, VEC_SIZE)
#define VSTORE_N CAT(vstore, VEC_SIZE)
#define INPUT_VEC_TYPE         CAT(INPUT0_TYPE, VEC_SIZE)
#define OUTPUT_VEC_TYPE        CAT(OUTPUT_TYPE, VEC_SIZE)
#define TO_OUTPUT_VEC_TYPE(x)  CAT(convert_, OUTPUT_VEC_TYPE)(x)

KERNEL(reorder_blocked_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT_REORDER_TYPE* input,
    __global OUTPUT_REORDER_TYPE* output
// [TEMP]
// #ifdef MEAN_SUBTRACT_IN_BUFFER
//     , __global MEAN_SUBTRACT_TYPE* mean_subtract
// #endif
    )
{
    const uint global_id = get_global_id(0);
    OUTPUT_VEC_TYPE res[ITEM_SIZE/4][4];

    unroll_for (uint i = 0; i < (ITEM_SIZE/4) ; ++i) {
        // OUTPUT_VEC_TYPE res = TO_OUTPUT_VEC_TYPE(VLOAD_N(global_id*ITEM_SIZE + i, input));
        res[i][0] = TO_OUTPUT_VEC_TYPE(VLOAD_N(global_id*ITEM_SIZE + i*2, input));
        res[i][1] = TO_OUTPUT_VEC_TYPE(VLOAD_N(global_id*ITEM_SIZE + i*2 + 1, input));
        res[i][2] = TO_OUTPUT_VEC_TYPE(VLOAD_N(global_id*ITEM_SIZE + i*4 + 2, input));
        res[i][3] = TO_OUTPUT_VEC_TYPE(VLOAD_N(global_id*ITEM_SIZE + i*4 + 3, input));

        // [TEMP]
        // res = ACTIVATION(res, ACTIVATION_PARAMS);

        // VSTORE_N(res, global_id * ITEM_SIZE + i, output);

        VSTORE_N(res[i][0], global_id*ITEM_SIZE + i*2, output);
        VSTORE_N(res[i][1], global_id*ITEM_SIZE + i*2 + 1, output);
        VSTORE_N(res[i][2], global_id*ITEM_SIZE + i*4 + 2, output);
        VSTORE_N(res[i][3], global_id*ITEM_SIZE + i*4 + 3, output);
    }
}
