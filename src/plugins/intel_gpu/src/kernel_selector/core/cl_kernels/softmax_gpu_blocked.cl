// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"


KERNEL(softmax)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint no_offset = 0;
    uint cls = 0;
    uint *b_offset, *f_offset, *z_offset, *y_offset, *x_offset;
    b_offset = f_offset = z_offset = y_offset = x_offset = &no_offset;

#if SOFTMAX_DIM_X
    x_offset = &cls;

    const uint b = get_global_id(2);
    const uint f = get_global_id(1);
    const uint z = (uint)get_global_id(0) % INPUT0_SIZE_Z;
    const uint y = (uint)get_global_id(0) / INPUT0_SIZE_Z;
    const uint x = 0;
#elif SOFTMAX_DIM_Y
    y_offset = &cls;

    const uint b = get_global_id(2);
    const uint f = get_global_id(1);
    const uint z = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint y = 0;
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
#elif SOFTMAX_DIM_Z
    z_offset = &cls;

    const uint b = get_global_id(2);
    const uint f = get_global_id(1);
    const uint z = 0;
    const uint y = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
#elif SOFTMAX_DIM_FEATURE
    f_offset = &cls;

    const uint b = get_global_id(2);
    const uint f = 0;
    const uint z = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint y = get_global_id(1);
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
#elif SOFTMAX_DIM_BATCH
    b_offset = &cls;

    const uint b = 0;
    const uint f = get_global_id(2);
    const uint z = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint y = get_global_id(1);
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
#else
#error
#endif

    ACCUMULATOR_TYPE max_value = UNIT_VAL_MIN;
    ACCUMULATOR_TYPE data[CLASS_NUM];

    for (cls = 0; cls < CLASS_NUM; ++cls)
    {
#if INPUT0_DIMS == 5
        const uint index = INPUT0_GET_INDEX(b + *b_offset, f + *f_offset, z + *z_offset, y + *y_offset, x + *x_offset);
#else
        const uint index = INPUT0_GET_INDEX(b + *b_offset, f + *f_offset, y + *y_offset, x + *x_offset);
#endif
    ACCUMULATOR_TYPE in = input[index];
    max_value = max(max_value, in);
    data[cls] = in;
}

    ACCUMULATOR_TYPE denominator = 0.0;
    for (cls = 0; cls < CLASS_NUM; ++cls)
    {
        data[cls] = native_exp(data[cls] - max_value);
        denominator += data[cls];
    }

    for (cls = 0; cls < CLASS_NUM; ++cls)
    {
        const ACCUMULATOR_TYPE res = data[cls] / denominator;
#if INPUT0_DIMS == 5
        const uint output_idx = OUTPUT_GET_INDEX(b + *b_offset, f + *f_offset, z + *z_offset, y + *y_offset, x + *x_offset);
#else
        const uint output_idx = OUTPUT_GET_INDEX(b + *b_offset, f + *f_offset, y + *y_offset, x + *x_offset);
#endif
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
    }
}
