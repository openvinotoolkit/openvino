// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

KERNEL(binary_convolution_ref)(const __global INPUT0_TYPE* input,
                                     __global OUTPUT_TYPE* output,
                               const __global FILTER_TYPE* weights,
#if HAS_FUSED_OPS_DECLS
                               FUSED_OPS_DECLS,
#endif
                               uint split_idx)
{
    const int b  = get_global_id(0);
    const int f  = get_global_id(1);
    const int yx = get_global_id(2);
    const int y = yx / OUTPUT_SIZE_X;
    const int x = yx % OUTPUT_SIZE_X;

    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const int output_index = OUTPUT_OFFSET
                           + b * OUTPUT_BATCH_PITCH
                           + f * OUTPUT_FEATURE_PITCH
                           + y * OUTPUT_Y_PITCH
                           + x * OUTPUT_X_PITCH;

    const int input_index = INPUT0_OFFSET
                          + b * INPUT0_FEATURE_NUM_PACKED*INPUT0_FEATURE_PITCH;

    const int weights_index = (f / OFM_BLOCK_SIZE) * INPUT0_FEATURE_NUM_PACKED*FILTER_SIZE_X*FILTER_SIZE_Y*OFM_BLOCK_SIZE
                            + (f % OFM_BLOCK_SIZE);
#if EXCLUDE_PAD
    int ks = 0;
#endif
    int res_popcnt = 0;
    for (int icp = 0; icp < INPUT0_FEATURE_NUM_PACKED; icp++)
    {
        for (int kh = 0; kh < FILTER_SIZE_Y; kh++)
        {
            const int input_offset_y = input_y + kh * DILATION_SIZE_Y;
            const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

            for (int kw = 0; kw < FILTER_SIZE_X; kw++)
            {
                const int input_offset_x = input_x + kw * DILATION_SIZE_X;
                const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;
                FILTER_TYPE wei = weights[weights_index + icp*OFM_BLOCK_SIZE*FILTER_SIZE_X*FILTER_SIZE_Y +
                                          kh*FILTER_SIZE_X*OFM_BLOCK_SIZE + kw*OFM_BLOCK_SIZE];
#if EXCLUDE_PAD
                if (!zero_y && !zero_x)
                {
                    INPUT0_TYPE src = input[input_index +
                                            icp * INPUT0_FEATURE_PITCH +
                                            input_offset_y*INPUT0_Y_PITCH +
                                            input_offset_x*INPUT0_X_PITCH]; // 32 packed input channels

#if LEFTOVERS
                    if (icp == INPUT0_FEATURE_NUM_PACKED - 1)
                        res_popcnt += popcount((src ^ wei) & LEFTOVERS_MASK);
                    else
#endif
                        res_popcnt += popcount(src ^ wei);
                    if (icp == 0)
                        ks++;
                }
#else
                if (zero_y || zero_x)
                {
#if LEFTOVERS
                    if (icp == INPUT0_FEATURE_NUM_PACKED - 1)
                        res_popcnt += popcount((PAD_VALUE ^ wei) & LEFTOVERS_MASK);
                    else
#endif
                        res_popcnt += popcount(PAD_VALUE ^ wei);
                }
                else
                {
                    INPUT0_TYPE src = input[input_index +
                                            icp * INPUT0_FEATURE_PITCH +
                                            input_offset_y*INPUT0_Y_PITCH +
                                            input_offset_x*INPUT0_X_PITCH]; // 32 packed input channels
#if LEFTOVERS
                    if (icp == INPUT0_FEATURE_NUM_PACKED - 1)
                        res_popcnt += popcount((src ^ wei) & LEFTOVERS_MASK);
                    else
#endif
                        res_popcnt += popcount(src ^ wei);

                }
#endif
            }
        }
    }


#if EXCLUDE_PAD
    UNIT_TYPE res = TO_OUTPUT_TYPE(INPUT0_FEATURE_NUM*ks - 2*res_popcnt);
#else
    UNIT_TYPE res = TO_OUTPUT_TYPE(INPUT0_FEATURE_NUM*FILTER_SIZE_X*FILTER_SIZE_Y - 2*res_popcnt);
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    res = FUSED_OPS_RESULT;
#endif

    output[output_index] = res;
}
