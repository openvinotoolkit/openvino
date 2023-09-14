// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

KERNEL(mha_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* inputq,
    const __global INPUT1_TYPE* inputk,
    const __global INPUT1_TYPE* inputv,
    __global OUTPUT_TYPE* output)
{
    /* TO BE FILLED */
}
