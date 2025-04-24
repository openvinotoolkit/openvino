// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

#include "include/reshape_dims.cl"

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
    uint8 ov = RESHAPE_DIMS(OUTPUT, INPUT0, b, f, 0, 0, w, z, y, x);
    const uint input_idx = FUNC_CALL(get_input_index)(ov.s0, ov.s1, ov.s2, ov.s3, ov.s4, ov.s5, ov.s6, ov.s7);
    const uint output_idx  = FUNC_CALL(get_output_index)(b, f, 0, 0, w, z, y, x);
#endif

#if   defined MEAN_SUBTRACT_INSIDE_PARAMS
    float res = TO_MEAN_TYPE(input[input_idx]);
    res -= VALUE_TO_SUBTRACT[f % VALUE_TO_SUBTRACT_SIZE];
#elif defined MEAN_SUBTRACT_IN_BUFFER
    MEAN_SUBTRACT_TYPE res = TO_MEAN_TYPE(input[input_idx]);
    uint8 msv = RESHAPE_DIMS(INPUT0, MEAN_SUBTRACT, b, f, 0, 0, w, z, y, x);
    res -= mean_subtract[GET_DATA_INDEX_SAFE(MEAN_SUBTRACT, msv.s0, msv.s1, msv.s6, msv.s7)];
#else
    CALC_TYPE res = TO_CALC_TYPE(input[input_idx]);
#endif

    output[output_idx] = ACTIVATION_TYPED(OUTPUT_REORDER, TO_OUTPUT_REORDER_TYPE_SAT(res), ACTIVATION_PARAMS_TYPED);
}
