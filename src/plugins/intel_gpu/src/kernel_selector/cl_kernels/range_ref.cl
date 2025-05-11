// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

KERNEL(range_ref)(OPTIONAL_SHAPE_INFO_ARG
                  const __global INPUT0_TYPE *startP,
                  const __global INPUT2_TYPE *stepP,
                  __global OUTPUT_TYPE *output)
{
    const uint i = get_global_id(2);
    const OUTPUT_TYPE start = TO_OUTPUT_TYPE(*startP);
    const OUTPUT_TYPE step = TO_OUTPUT_TYPE(*stepP);
    output[i] = start + TO_OUTPUT_TYPE(i) * step;
}
