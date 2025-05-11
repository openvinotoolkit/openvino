// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#if !IS_DYNAMIC
REQD_SUB_GROUP_SIZE(16)
#endif
KERNEL(softmax)(
    OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input,
#ifdef IS_DYNAMIC
    __global ACCUMULATOR_TYPE* tmp_buffer,
#endif
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    uint cls = 0;
#if INPUT0_SIMPLE == 1
#if INPUT0_DIMS == 5
    const uint other0 = (uint)get_global_id(0) % INPUT0_OTHER0_SIZE;
    const uint other2 = (uint)get_global_id(0) / INPUT0_OTHER0_SIZE;
#else
    const uint other0 = get_global_id(0);
    const uint other2 = 0;
#endif
    const uint other1 = get_global_id(1);
    const uint other3  = get_global_id(2);

    const uint in_depth_offset  = other3*INPUT0_OTHER3_PITCH + other2*INPUT0_OTHER2_PITCH + other1*INPUT0_OTHER1_PITCH + other0*INPUT0_OTHER0_PITCH + INPUT0_OFFSET;
    const uint out_depth_offset = other3*OUTPUT_OTHER3_PITCH + other2*OUTPUT_OTHER2_PITCH + other1*OUTPUT_OTHER1_PITCH + other0*OUTPUT_OTHER0_PITCH + OUTPUT_OFFSET;
#else // blocked format
    const uint no_offset = 0;
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
#error Wrong axis
#endif
#endif

    const size_t class_num = INPUT0_CLASS_NUM;
    ACCUMULATOR_TYPE max_value = UNIT_VAL_MIN;
#if IS_DYNAMIC
    #define TMP_CLASS_PITCH INPUT0_CLASS_PITCH
    __global ACCUMULATOR_TYPE* data = tmp_buffer + in_depth_offset;
#else
    #define TMP_CLASS_PITCH 1
    ACCUMULATOR_TYPE data[INPUT0_CLASS_NUM];
#endif
    for (cls = 0; cls < INPUT0_CLASS_NUM; ++cls)
    {
#if INPUT0_SIMPLE == 1
        const uint index = in_depth_offset + cls*INPUT0_CLASS_PITCH;
#else
#if INPUT0_DIMS == 5
        const uint index = INPUT0_GET_INDEX(b + *b_offset, f + *f_offset, z + *z_offset, y + *y_offset, x + *x_offset);
#else
        const uint index = INPUT0_GET_INDEX(b + *b_offset, f + *f_offset, y + *y_offset, x + *x_offset);
#endif
#endif
        ACCUMULATOR_TYPE in = input[index];
        max_value = max(max_value, in);
        data[cls*TMP_CLASS_PITCH] = in;
    }

    // TODO: currently we calculate on float32 because it's lot of "add" operation and it stuck on the value "8192.0f"
    ACCUMULATOR_TYPE denominator = 0.0;
    for (cls = 0; cls < class_num; ++cls) {
        ACCUMULATOR_TYPE t = native_exp(data[cls*TMP_CLASS_PITCH] - max_value);
        denominator += t;
        data[cls*TMP_CLASS_PITCH] = t;
    }

    for (cls = 0; cls < class_num; ++cls) {
        const ACCUMULATOR_TYPE res = data[cls*TMP_CLASS_PITCH] / denominator;
#if INPUT0_SIMPLE == 1
        const uint output_idx = out_depth_offset + cls*OUTPUT_CLASS_PITCH;
#else
#if INPUT0_DIMS == 5
        const uint output_idx = OUTPUT_GET_INDEX(b + *b_offset, f + *f_offset, z + *z_offset, y + *y_offset, x + *x_offset);
#else
        const uint output_idx = OUTPUT_GET_INDEX(b + *b_offset, f + *f_offset, y + *y_offset, x + *x_offset);
#endif
#endif
#if HAS_FUSED_OPS
        FUSED_OPS;
        output[output_idx] = FUSED_OPS_RESULT;
#else
        output[output_idx] = ACTIVATION(res, ACTIVATION_PARAMS);
#endif
    }
}
