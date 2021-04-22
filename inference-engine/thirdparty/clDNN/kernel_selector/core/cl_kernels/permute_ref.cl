// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/include_all.cl"

KERNEL (permute_ref)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    //gws(x, y * z * w, b*f)
    const uint gid_0 = get_global_id(1);
#if INPUT0_DIMS == 4
    const uint y = gid_0;
#elif INPUT0_DIMS == 5
    const uint z = gid_0 / INPUT0_SIZE_Y;
    const uint y = gid_0 % INPUT0_SIZE_Y;   
#else
    const uint w = gid_0 / (INPUT0_SIZE_Y * INPUT0_SIZE_Z) % INPUT0_SIZE_W;
    const uint z = gid_0 / INPUT0_SIZE_Y % INPUT0_SIZE_Z;
    const uint y = gid_0 % INPUT0_SIZE_Y;
#endif

    const uint x = get_global_id(0);
    const uint f = (uint)get_global_id(2) % INPUT0_FEATURE_NUM;
    const uint b = (uint)get_global_id(2) / INPUT0_FEATURE_NUM;

    INPUT0_TYPE input_var = input[IN_IDX];

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[OUT_IDX] = FUSED_OPS_RESULT;
#else
    output[OUT_IDX] = ACTIVATION(input[IN_IDX], ACTIVATION_PARAMS);
#endif
}
