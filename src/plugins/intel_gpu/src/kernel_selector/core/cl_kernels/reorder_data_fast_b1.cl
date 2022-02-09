// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/data_types.cl"
#include "include/reshape_dims.cl"


///////////////////////// Input Index /////////////////////////
inline uint FUNC(get_input_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if INPUT0_DIMS < 5
    return INPUT0_GET_INDEX(b, f, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 5
    return GET_DATA_INDEX_5D(INPUT0, b, f, z, y, x);
#elif INPUT0_SIMPLE && INPUT0_DIMS == 6
    return GET_DATA_INDEX_6D(INPUT0, b, f, w, z, y, x);
#elif defined INPUT0_LAYOUT_BS_F_BSV8__AF8  || \
      defined INPUT0_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(INPUT0, b, f, y, x, SUB_GROUP_SIZE);
#elif defined INPUT0_LAYOUT_B_FS_YX_FSV16
    return GET_DATA_B_FS_YX_FSV16_INDEX(INPUT0, b, f, y, x);
#elif defined INPUT0_LAYOUT_B_FS_ZYX_FSV16
    return GET_DATA_B_FS_ZYX_FSV16_INDEX(INPUT0, b, f, z, y, x);
#else
#error reorder_data_fast_b1.cl: input format - not supported
#endif
}

///////////////////////// Output Index /////////////////////////

inline uint FUNC(get_output_index)(uint b, uint f, uint w, uint z, uint y, uint x)
{
#if OUTPUT_DIMS < 5
    return OUTPUT_GET_INDEX(b, f, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 5
    return GET_DATA_INDEX_5D(OUTPUT, b, f, z, y, x);
#elif OUTPUT_SIMPLE && OUTPUT_DIMS == 6
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, y, x);
#elif defined OUTPUT_LAYOUT_BS_F_BSV8__AF8  || \
      defined OUTPUT_LAYOUT_BS_F_BSV16__AF8
    return GET_DATA_BS_FYX_BSV8_INDEX(OUTPUT, b, f, y, x, SUB_GROUP_SIZE);
#elif defined OUTPUT_LAYOUT_B_FS_YX_FSV16
    return GET_DATA_B_FS_YX_FSV16_INDEX(OUTPUT, b, f, y, x);
#elif defined OUTPUT_LAYOUT_B_FS_ZYX_FSV16
    return GET_DATA_B_FS_ZYX_FSV16_INDEX(OUTPUT, b, f, z, y, x);
#else
#error reorder_data_fast_b1.cl: output format - not supported
#endif
}

KERNEL (reorder_data_fast_b1)(
    const __global INPUT_REORDER_TYPE* input,
    __global OUTPUT_REORDER_TYPE* output
#ifdef MEAN_SUBTRACT_IN_BUFFER
    , __global MEAN_SUBTRACT_TYPE* mean_subtract
#endif
    )
{
    uint data_idx = get_global_id(0);
    if(data_idx >= ELEMENTS_COUNT)
        return;

#if !CHANGE_DATA_TYPE_ONLY
 // We're checking output layout instead of input layout intentionally for performance reason
#if defined OUTPUT_LAYOUT_BFYX
    uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    data_idx = tmp_data_idx;

    // We're first iterating over Y then over X for performance reason
    // Otherwise we could compute X and Y in reverse order
    tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
    const uint z = 0;
    const uint w = 0;
#elif defined OUTPUT_LAYOUT_YXFB
    // We're first iterating over Y then over X for performance reason
    // Otherwise we could compute X and Y in reverse order
    uint tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    const uint z = 0;
    const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFYX_8F
    uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    data_idx = tmp_data_idx;

    // We're first iterating over Y then over X for performance reason
    // Otherwise we could compute X and Y in reverse order
    tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
    const uint z = 0;
    const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFYX_16F
    uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    data_idx = tmp_data_idx;

    // We're first iterating over Y then over X for performance reason
    // Otherwise we could compute X and Y in reverse order
    tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
    const uint z = 0;
    const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFZYX
    uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;

    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Z;
    const uint z = data_idx - tmp_data_idx * OUTPUT_SIZE_Z;
    const uint w = 0;
#elif defined OUTPUT_LAYOUT_B_FS_ZYX_FSV16
    uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Z;
    const uint z = data_idx - tmp_data_idx * OUTPUT_SIZE_Z;
    const uint w = 0;
#elif defined OUTPUT_LAYOUT_BFWZYX
    uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_Z;
    const uint z = data_idx - tmp_data_idx * OUTPUT_SIZE_Z;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_SIZE_W;
    const uint w = data_idx - tmp_data_idx * OUTPUT_SIZE_W;
#else // BYXF?
    uint tmp_data_idx = data_idx / OUTPUT_BATCH_NUM;
    const uint b = data_idx - tmp_data_idx * OUTPUT_BATCH_NUM;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_SIZE_Y;
    const uint y = data_idx - tmp_data_idx * OUTPUT_SIZE_Y;
    data_idx = tmp_data_idx;

    tmp_data_idx = data_idx / OUTPUT_SIZE_X;
    const uint x = data_idx - tmp_data_idx * OUTPUT_SIZE_X;
    data_idx = tmp_data_idx;

    tmp_data_idx  = data_idx / OUTPUT_FEATURE_NUM;
    const uint f = data_idx - tmp_data_idx * OUTPUT_FEATURE_NUM;
    const uint z = 0;
    const uint w = 0;
#endif
#endif

#if CHANGE_DATA_TYPE_ONLY
    const uint input_idx  = data_idx;
    const uint output_idx = data_idx;
#else
    uint8 ov = RESHAPE_DIMS(OUTPUT, INPUT0, b, f, w, z, y, x);
    const uint input_idx = FUNC_CALL(get_input_index)(ov[1], ov[2], ov[3], ov[4], ov[5],ov[6]);
    const uint output_idx  = FUNC_CALL(get_output_index)(b, f, w, z, y, x);
#endif

#if   defined MEAN_SUBTRACT_INSIDE_PARAMS
    float res = TO_MEAN_TYPE(input[input_idx]);
    res -= VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE];
#elif defined MEAN_SUBTRACT_IN_BUFFER
    MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
    uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, w, z, y, x);
    res -= mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv[1], msv[2], msv[5], msv[6])];
#else
    CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
#endif

    output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res), ACTIVATION_PARAMS_TYPED);
}
