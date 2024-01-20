// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(swiglu_gpu_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output)
{
    // TO_BE_IMPLEMENTED
}
