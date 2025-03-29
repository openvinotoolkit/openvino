// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

KERNEL(normalization)(
    __global const INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint b = get_global_id(GWS_BATCH);
    const uint f = get_global_id(GWS_FEATURE);
    const uint y = (uint)get_global_id(GWS_YX) / INPUT0_SIZE_X;
    const uint x = (uint)get_global_id(GWS_YX) % INPUT0_SIZE_X;

    const uint input_index  = GET_DATA_INDEX(INPUT0, b, f, y, x);
    const uint output_index = GET_DATA_INDEX(OUTPUT, b, f, y, x);

    ACCUMULATOR_TYPE sum = 0.0f;
#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

#ifdef ACROSS_CHANNEL

    uint j_offset = input_index - PADDING*INPUT0_FEATURE_PITCH;

    for(int j = 0 ; j < LOCAL_SIZE ; j++)
    {
        const int z_idx = (j + f - PADDING);
        bool zero = (z_idx < 0 || z_idx >= INPUT0_FEATURE_NUM);
        INPUT0_TYPE val = zero ? 0.0f : input[j_offset];
        sum += val*val;
        j_offset += INPUT0_FEATURE_PITCH;
#ifdef DYNAMIC_KERNEL_DIVIDER
        num_elementes += zero ? 0 : 1;
#endif
    }

#else

    const int x_start = ((int)x - PADDING);
    const int y_start = ((int)y - PADDING);
    int input_offset = GET_DATA_INDEX(INPUT0, b, f, y_start, x_start);

    for (int j = 0; j < LOCAL_SIZE ; ++j)
    {
        for (int i = 0; i < LOCAL_SIZE ; ++i)
        {
            int input_offset_x = x_start + i;
            int input_offset_y = y_start + j;
            bool zero = false;
            zero = input_offset_x < 0 ? true : zero;
            zero = input_offset_y < 0 ? true : zero;
            zero = input_offset_x >= INPUT0_SIZE_X ? true : zero;
            zero = input_offset_y >= INPUT0_SIZE_Y ? true : zero;

            INPUT0_TYPE val = zero ? INPUT0_VAL_ZERO : input[input_offset];

            sum += val*val;
            input_offset += INPUT0_X_PITCH;
#ifdef DYNAMIC_KERNEL_DIVIDER
            num_elementes += zero ? 0 : 1;
#endif
        }
        input_offset += INPUT0_Y_PITCH - LOCAL_SIZE*INPUT0_X_PITCH;
    }
#endif

#ifdef DYNAMIC_KERNEL_DIVIDER
    const INPUT0_TYPE num_elementes_div = INPUT0_VAL_ONE / TO_INPUT0_TYPE(num_elementes);
#else
    const INPUT0_TYPE num_elementes_div = NUM_ELEMENTS_DIV;
#endif

    INPUT0_TYPE base = TO_INPUT0_TYPE(K) + TO_INPUT0_TYPE((ACCUMULATOR_TYPE)ALPHA*sum * num_elementes_div);
    INPUT0_TYPE normalization_factor = native_powr(base, TO_INPUT0_TYPE(-BETA));

    INPUT0_TYPE lrn_result = input[input_index] * normalization_factor;

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[output_index] = res;
#else
    output[output_index] = ACTIVATION(lrn_result, ACTIVATION_PARAMS);
#endif

}
