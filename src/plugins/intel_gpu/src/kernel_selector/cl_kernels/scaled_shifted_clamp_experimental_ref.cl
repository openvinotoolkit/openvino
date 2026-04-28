// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

KERNEL(scaled_shifted_clamp_experimental_ref)(OPTIONAL_SHAPE_INFO_ARG
                                              const __global INPUT0_TYPE* input,
                                              __global OUTPUT_TYPE* output)
{
    const uint i = get_global_id(2);
    OUTPUT_TYPE y = TO_OUTPUT_TYPE(input[i]) * TO_OUTPUT_TYPE(SCALE) + TO_OUTPUT_TYPE(BIAS);
    y = max(y, TO_OUTPUT_TYPE(LO));
    y = min(y, TO_OUTPUT_TYPE(HI));
    output[i] = y;
}
