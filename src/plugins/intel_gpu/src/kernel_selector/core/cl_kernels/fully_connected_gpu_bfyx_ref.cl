// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

KERNEL(fc)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
#if OUTPUT_3D
    const uint oxfm = get_global_id(0);
    const uint b = get_global_id(1);
    const uint oym = oxfm % OUTPUT_SIZE_Y;
    const uint ofm = oxfm / OUTPUT_SIZE_Y;

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint y = 0; y < INPUT0_SIZE_Y; ++y)
    {
        for(uint x = 0; x < INPUT0_SIZE_X; ++x )
        {
            const uint input0_idx = GET_DATA_INDEX(INPUT0, b, ofm, y, x);
            const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, oym, y, 0, 0);
            dotProd += (ACCUMULATOR_TYPE)(input[input0_idx] * weights[filter_idx]);
        }
    }

    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, ofm, oym, 0);
    const uint bias_index = oym;
#else
    const uint ofm = get_global_id(0);
    const uint b = get_global_id(1);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint ifm = 0; ifm < INPUT0_FEATURE_NUM; ++ifm)
    {
       for (uint y = 0; y < INPUT0_SIZE_Y; ++y)
       {
           for(uint x = 0; x < INPUT0_SIZE_X; ++x )
           {
               const uint input0_idx = GET_DATA_INDEX(INPUT0, b, ifm, y, x);
               const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, ofm, ifm, y, x);
               dotProd += (ACCUMULATOR_TYPE)(input[input0_idx] * weights[filter_idx]);
          }
       }
    }

    const uint dst_index = GET_DATA_INDEX(OUTPUT, b, ofm, 0, 0);
    const uint bias_index = ofm;
#endif

#if BIAS_TERM
    ACTIVATION_TYPE dequantized = dotProd + biases[bias_index];
#else
    ACTIVATION_TYPE dequantized = dotProd;
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[dst_index] = res;
#else
    output[dst_index] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(dequantized, ACTIVATION_PARAMS_TYPED));
#endif

}
