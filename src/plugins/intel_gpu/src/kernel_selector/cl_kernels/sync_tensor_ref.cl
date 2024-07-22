// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#if !IS_DYNAMIC
REQD_SUB_GROUP_SIZE(16)
#endif
KERNEL(sync_tensor)(
    OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input,
// #ifdef IS_DYNAMIC
//     __global ACCUMULATOR_TYPE* tmp_buffer,
// #endif
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    printf("[kernel] sync_tensor_ref.cl sync_tensor kernel\n");
    const uint b = get_global_id(0);
    const uint p = get_global_id(1);
}
