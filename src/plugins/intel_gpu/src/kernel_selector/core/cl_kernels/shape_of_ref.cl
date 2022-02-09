// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"

KERNEL(shape_of_ref)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
    )
{
    const unsigned int i = (uint)get_global_id(2);

    output[i] = INPUT0_SIZES[INPUT_RANK - 1 - i];
}
