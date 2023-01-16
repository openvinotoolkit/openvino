// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(shape_of_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    __global OUTPUT_TYPE* output
    )
{
    const unsigned int i = (uint)get_global_id(2);

#if IS_DYNAMIC
    output[i] = TO_OUTPUT_TYPE(shape_info[i]);
#else
    size_t shapes[] = INPUT_DIMS_INIT;
    output[i] = TO_OUTPUT_TYPE(shapes[i]);
#endif
}
