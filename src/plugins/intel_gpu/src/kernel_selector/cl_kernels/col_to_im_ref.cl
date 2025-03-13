// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(col_to_im_ref)(const __global INPUT0_TYPE* input,
                                 __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                           , FUSED_OPS_DECLS
#endif
)
{
}