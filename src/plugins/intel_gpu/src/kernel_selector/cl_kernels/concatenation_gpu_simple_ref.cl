// Copyright (C) 2018-2026 Intel Corporation
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
    uint gid1 = (uint)get_global_id(1);
    const uint z = gid1 % INPUT0_SIZE_Z;
    gid1 /= INPUT0_SIZE_Z;
    const uint w = gid1 % INPUT0_SIZE_W;
    gid1 /= INPUT0_SIZE_W;
    const uint u = gid1 % INPUT0_SIZE_U;
    gid1 /= INPUT0_SIZE_U;
    const uint v = gid1;
    const uint f = (uint)get_global_id(2) % INPUT0_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / INPUT0_FEATURE_NUM;

    uint out_x = x;
    uint out_y = y;
    uint out_z = z;
    uint out_w = w;
    uint out_u = u;
    uint out_v = v;
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
#elif CONCAT_U
    out_u += output_offset_in_concat_axis;
#elif CONCAT_V
    out_v += output_offset_in_concat_axis;
#elif CONCAT_FEATURE
    out_f += output_offset_in_concat_axis;
#elif CONCAT_BATCH
    out_b += output_offset_in_concat_axis;
#else
#   error concatenation_gpu_bfzyx_ref.cl: Unrecognized concat axis.
#endif

    uint input_offset  = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, v, u, w, z, y, x);
    uint output_offset = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR out_b, out_f, out_v, out_u, out_w, out_z, out_y, out_x);

    INPUT0_TYPE result = input[input_offset];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_offset] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#endif
}
