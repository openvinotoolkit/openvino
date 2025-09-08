// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/reshape_dims.cl"
#include "include/batch_headers/fetch_data.cl"



inline void FUNC(get_yxfb_coords_from_linear_idx_no_padding)(uint data_idx, uint* b, uint* f, uint* x, uint* y)
{
    uint tmp_data_idx = data_idx / INPUT0_BATCH_NUM;
    *b = data_idx - tmp_data_idx * INPUT0_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_FEATURE_NUM;
    *f = data_idx - tmp_data_idx * INPUT0_FEATURE_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / INPUT0_SIZE_X;
    *x = data_idx - tmp_data_idx * INPUT0_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / INPUT0_SIZE_Y;
    *y = data_idx - tmp_data_idx * INPUT0_SIZE_Y;
}

REQD_SUB_GROUP_SIZE(8)
KERNEL (reorder_data_to_yxfb_batched)(
    const __global INPUT_REORDER_TYPE* input,
    __global OUTPUT_REORDER_TYPE* output
    #ifdef MEAN_SUBTRACT_IN_BUFFER
    , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
    )
{
    uint group_idx = (uint)get_group_id(0) * OUTPUT_BATCH_NUM * 8;

    for(uint i = 0; i < OUTPUT_BATCH_NUM; i++)
    {
        uint output_idx = group_idx + (uint)get_sub_group_local_id();
        if(output_idx >= ELEMENTS_COUNT)
            continue;

        group_idx += 8;

        uint x,y,f,b;
        FUNC_CALL(get_yxfb_coords_from_linear_idx_no_padding)(output_idx, &b,&f,&x,&y);
        const uint input_idx = INPUT0_GET_INDEX(b, f, y, x);

    #if defined MEAN_SUBTRACT_INSIDE_PARAMS
        float res = TO_MEAN_TYPE(input[input_idx]);
        res = MEAN_OP(res, VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE]);
    #elif defined MEAN_SUBTRACT_IN_BUFFER
    #if defined MEAN_PER_FEATURE
        MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
        res = MEAN_OP(res, mean_subtract[f]);
    #else
        MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
        uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, 0, 0, y,x);
        res = MEAN_OP(res, mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[1], msv[2], msv[5], msv[6])]);
    #endif
    #else
        CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
    #endif

        output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res), ACTIVATION_PARAMS_TYPED);
    }
}
