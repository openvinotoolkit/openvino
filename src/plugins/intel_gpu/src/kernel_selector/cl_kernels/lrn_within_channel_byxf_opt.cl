// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/acc_type.cl"

#define VECTOR_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#define ACCUMULATOR_VECTOR_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#define FEATURE_PER_ITEM 8
#define FEATURE_BLOCK_NUM (OUTPUT_FEATURE_NUM / 8)

KERNEL(lrn_within_channel_byxf_opt)(
    __global const INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
    const uint b = get_global_id(GWS_BATCH);
    const uint f = (uint)get_global_id(GWS_FEATURE)*FEATURE_PER_ITEM;
    const uint y = (uint)get_global_id(GWS_YX) / INPUT0_SIZE_X;
    const uint x = (uint)get_global_id(GWS_YX) % INPUT0_SIZE_X;

    const uint input_index = GET_DATA_INDEX(INPUT0, b, f, y, x);
    const uint output_index = GET_DATA_INDEX(OUTPUT, b, f, y, x);

    ACCUMULATOR_VECTOR_TYPE sum = 0.0f;
#ifdef DYNAMIC_KERNEL_DIVIDER
    uint num_elementes = 0;
#endif

    const int x_start = ((int)x - PADDING);
    const int y_start = ((int)y - PADDING);
    int input_offset = ((int)(GET_DATA_INDEX(INPUT0, b, f, y_start, x_start)))/8;

    VECTOR_TYPE feature_block;

    for (int j = 0; j < LOCAL_SIZE; ++j)
    {
        for (int i = 0; i < LOCAL_SIZE; ++i)
        {
            int input_offset_x = x_start + i;
            int input_offset_y = y_start + j;
            bool zero = false;
            zero = input_offset_x < 0 ? true : zero;
            zero = input_offset_y < 0 ? true : zero;
            zero = input_offset_x >= INPUT0_SIZE_X ? true : zero;
            zero = input_offset_y >= INPUT0_SIZE_Y ? true : zero;

            VECTOR_TYPE val = zero ? INPUT0_VAL_ZERO : vload8(input_offset+FEATURE_BLOCK_NUM*i, input);

            sum = mad(val,val,sum);
#ifdef DYNAMIC_KERNEL_DIVIDER
            num_elementes += zero ? 0 : 1;
#endif
        }
        input_offset += INPUT0_Y_PITCH/FEATURE_PER_ITEM;
    }

#ifdef DYNAMIC_KERNEL_DIVIDER
    const INPUT0_TYPE num_elementes_div = INPUT0_VAL_ONE / TO_INPUT0_TYPE(num_elementes);
#else
    const INPUT0_TYPE num_elementes_div = NUM_ELEMENTS_DIV;
#endif

    const VECTOR_TYPE base = mad((ACCUMULATOR_TYPE)ALPHA*num_elementes_div, sum, TO_INPUT0_TYPE(K));
    const VECTOR_TYPE normalization_factor = native_powr(base, TO_INPUT0_TYPE(-BETA));
    const VECTOR_TYPE val = vload8(input_index/FEATURE_PER_ITEM, input);
    const VECTOR_TYPE normes = val*normalization_factor;

    INPUT0_TYPE lrn_result;

    for(uint i = 0; i < FEATURE_PER_ITEM; i++)
    {
        lrn_result = normes[i];
        #if HAS_FUSED_OPS
            FUSED_OPS;
            OUTPUT_TYPE res = FUSED_OPS_RESULT;
            output[output_index+i] = res;
        #else
            output[output_index+i] = ACTIVATION(lrn_result, ACTIVATION_PARAMS);
        #endif
    }
}

#undef FEATURE_BLOCK_NUM
#undef FEATURE_PER_ITEM
#undef VECTOR_TYPE
#undef ACCUMULATOR_VECTOR_TYPE
