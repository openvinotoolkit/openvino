// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

KERNEL(convolution_mmad_b_fs_yx_fsv32_dw)(
    __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    __global FILTER_TYPE* weights
#if BIAS_TERM
    , __global BIAS_TYPE* biases
#endif
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    , const __global WEIGHTS_ZERO_POINTS_TYPE *weights_zp
#endif
#if ASYMMETRIC_DATA_QUANTIZATION
    , const __global ACTIVATIONS_ZERO_POINTS_TYPE *activations_zp
#endif
#if COMPENSATION_TERM
    , const __global COMPENSATION_TYPE *compensation
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint f = get_global_id(0);
    const uint b = get_global_id(2);
    const uint x = get_global_id(1) % OUTPUT_SIZE_X;
    const uint y = get_global_id(1) / OUTPUT_SIZE_X;
    const uint g = f;

    int dotProd = 0;
    const int input_x = x * STRIDE_SIZE_X - PADDING_SIZE_X;
    const int input_y = y * STRIDE_SIZE_Y - PADDING_SIZE_Y;

    const uint filter_offset = g * FILTER_GROUPS_PITCH;

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    int src_sum = 0;
#endif

    for (uint k = 0; k < FILTER_IFM_NUM; ++k)
    {
        for (uint j = 0; j < FILTER_SIZE_Y ; ++j)
        {
            const int input_offset_y = input_y + j * DILATION_SIZE_Y;
            const bool zero_y = input_offset_y >= INPUT0_SIZE_Y || input_offset_y < 0;

            for (uint i = 0; i < FILTER_SIZE_X ; ++i)
            {
                const int input_offset_x = input_x + i * DILATION_SIZE_X;
                const bool zero_x = input_offset_x >= INPUT0_SIZE_X || input_offset_x < 0;

                uint input_idx = INPUT0_GET_INDEX(b, g*FILTER_IFM_NUM + k, input_offset_y, input_offset_x);
                int in = 0;
                if (!zero_y && !zero_x)
                    in = input[input_idx];
#if ASYMMETRIC_DATA_QUANTIZATION
                else
                    in = activations_zp[g*FILTER_IFM_NUM + k];
#endif

                uint filter_idx = filter_offset + k*FILTER_IFM_PITCH + j*FILTER_Y_PITCH + i*FILTER_X_PITCH;

                dotProd += in * (int)weights[filter_idx];
#if ASYMMETRIC_WEIGHTS_QUANTIZATION
                src_sum += in;
#endif
            }
        }
    }

#if BIAS_TERM
    float res = (float)dotProd + biases[f];
#else
    float res = (float)dotProd;
#endif

#if ASYMMETRIC_WEIGHTS_QUANTIZATION
    res -= src_sum * weights_zp[f];
#endif

#if COMPENSATION_TERM
    res += compensation[f];
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE out = FUSED_OPS_RESULT;
#else
    OUTPUT_TYPE out = TO_OUTPUT_TYPE(res);
#endif

    const uint dst_index = OUTPUT_GET_INDEX(b, f, y, x);

    output[dst_index] = ACTIVATION(out, ACTIVATION_PARAMS);
}
