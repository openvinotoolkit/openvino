// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

KERNEL(grn_ref)(
    const __global INPUT0_TYPE*  input,
    __global OUTPUT_TYPE* output)
{
    const uint ob = get_global_id(0);
    const uint oy = get_global_id(1);
    const uint ox = get_global_id(2);

    int in_offset  = INPUT0_GET_INDEX(ob, 0, oy, ox);
    int out_offset = OUTPUT_GET_INDEX(ob, 0, oy, ox);

    ACCUMULATOR_TYPE variance = 0;
    for (int f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        int in_off = in_offset + f * INPUT0_FEATURE_PITCH;
        INPUT0_TYPE val = input[in_off];
        variance += val*val;
    }
    variance = sqrt(variance + BIAS);
    for (int f = 0; f < INPUT0_FEATURE_NUM; f++)
    {
        int in_off = in_offset + f * INPUT0_FEATURE_PITCH;
        int out_off = out_offset + f * OUTPUT_FEATURE_PITCH;
        output[out_off] = (OUTPUT_TYPE)(input[in_off] / variance);
    }
}
