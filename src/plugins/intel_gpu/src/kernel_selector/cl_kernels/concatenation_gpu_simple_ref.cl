// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

KERNEL (concatenation_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    uint output_offset_in_concat_axis
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint x = (uint)get_global_id(0) % INPUT0_SIZE_X;
    const uint y = (uint)get_global_id(0) / INPUT0_SIZE_X;
    const uint z = (uint)get_global_id(1) % INPUT0_SIZE_Z;
    const uint w = (uint)get_global_id(1) / INPUT0_SIZE_Z;
    const uint f = (uint)get_global_id(2) % INPUT0_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / INPUT0_FEATURE_NUM;

    uint out_x = x;
    uint out_y = y;
    uint out_z = z;
    uint out_w = w;
    uint out_f = f;
    uint out_b = b;

#if CONCAT_X
    out_x += output_offset_in_concat_axis;
#elif CONCAT_Y
    out_y += output_offset_in_concat_axis;
#elif CONCAT_Z
    out_z += output_offset_in_concat_axis;
#elif CONCAT_W
    out_w += output_offset_in_concat_axis;
#elif CONCAT_FEATURE
    out_f += output_offset_in_concat_axis;
#elif CONCAT_BATCH
    out_b += output_offset_in_concat_axis;
#else
#   error concatenation_gpu_bfzyx_ref.cl: Unrecognized concat axis.
#endif

    uint input_offset  = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x);
    uint output_offset = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_w, out_z, out_y, out_x);

    INPUT0_TYPE result = input[input_offset];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_offset] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#endif
    // if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
    //     printf("mingyuki: log from concate_gpu_simple_ref: %d,%d,%d,%d -> %d,%d,%d,%d     offset %d->%d\n", INPUT0_BATCH_NUM, INPUT0_FEATURE_NUM, INPUT0_SIZE_Y, INPUT0_SIZE_X, OUTPUT_BATCH_NUM, OUTPUT_FEATURE_NUM, OUTPUT_SIZE_Y, OUTPUT_SIZE_X,
    //         input_offset, output_offset);
    //     printf("mingyuki: first value %d, sizeof first elem %d  --> output %d\n", input[0], sizeof(input[0]), (int)output[output_offset]);
    // }
}
