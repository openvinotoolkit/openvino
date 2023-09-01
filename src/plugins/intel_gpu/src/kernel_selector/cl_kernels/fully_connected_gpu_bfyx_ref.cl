// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"

KERNEL(fc)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    , const __global BIAS_TYPE* biases
#endif
#if DECOMPRESSION_SCALE_TERM
    , const __global DECOMPRESSION_SCALE_TYPE* decompression_scale
#endif
#if DECOMPRESSION_ZP_TERM
    , const __global DECOMPRESSION_ZP_TYPE* decompression_zp
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    )
{
#if OUTPUT_3D
    const uint ofm = get_global_id(0);
    const uint oym = get_global_id(1);
    const uint b = get_global_id(2);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint y = 0; y < INPUT0_SIZE_Y; ++y)
    {
        for (uint x = 0; x < INPUT0_SIZE_X; ++x)
        {
            const uint input0_idx = INPUT0_GET_INDEX(b, ofm, y, x);
            const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, oym, y, 0, 0);
            #if COMPRESSED_WEIGHTS
                ACCUMULATOR_TYPE filter_compressed = TO_ACCUMULATOR_TYPE(weights[filter_idx]);
                #if DECOMPRESSION_ZP_TERM
                    ACCUMULATOR_TYPE zp = TO_ACCUMULATOR_TYPE(decompression_zp[DECOMPRESSION_ZP_GET_INDEX_SAFE(0, oym, 0, 0)]);
                #else
                    ACCUMULATOR_TYPE zp = ACCUMULATOR_VAL_ZERO;
                #endif
                DECOMPRESSION_SCALE_TYPE scale = decompression_scale[DECOMPRESSION_SCALE_GET_INDEX_SAFE(0, oym, 0, 0)];
                ACCUMULATOR_TYPE filter_val = (TO_ACCUMULATOR_TYPE(filter_compressed) - TO_ACCUMULATOR_TYPE(zp)) * scale;
                dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * (ACCUMULATOR_TYPE)(filter_val);
            #else
                dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * (ACCUMULATOR_TYPE)(weights[filter_idx]);
            #endif
        }
    }

    const uint dst_index = OUTPUT_GET_INDEX(b, ofm, oym, 0);
#else
    const uint ofm = get_global_id(0);
    const uint b = get_global_id(1);

    ACCUMULATOR_TYPE dotProd = ACCUMULATOR_VAL_ZERO;

    for (uint ifm = 0; ifm < INPUT0_FEATURE_NUM; ++ifm)
    {
        for (uint y = 0; y < INPUT0_SIZE_Y; ++y)
        {
           for (uint x = 0; x < INPUT0_SIZE_X; ++x)
            {
                const uint input0_idx = INPUT0_GET_INDEX(b, ifm, y, x);
                const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, ofm, ifm, y, x);
                #if COMPRESSED_WEIGHTS
                    FILTER_TYPE filter_compressed = weights[filter_idx];
                    #if DECOMPRESSION_ZP_TERM
                        ACCUMULATOR_TYPE zp = decompression_zp[DECOMPRESSION_ZP_GET_INDEX_SAFE(0, ofm, 0, 0)];
                    #else
                        ACCUMULATOR_TYPE zp = ACCUMULATOR_VAL_ZERO;
                    #endif

                    DECOMPRESSION_SCALE_TYPE scale = decompression_scale[DECOMPRESSION_SCALE_GET_INDEX_SAFE(0, ofm, 0, 0)];
                    ACCUMULATOR_TYPE filter_val = (TO_ACCUMULATOR_TYPE(filter_compressed) - TO_ACCUMULATOR_TYPE(zp)) * scale;
                    dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * (ACCUMULATOR_TYPE)(filter_val);
                #else
                    dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * (ACCUMULATOR_TYPE)(weights[filter_idx]);
                #endif
            }
        }
    }

    const uint dst_index = OUTPUT_GET_INDEX(b, ofm, 0, 0);
#endif

#if BIAS_TERM
    #if BIAS_PER_OUTPUT
        #if OUTPUT_3D
            const uint bias_index = GET_DATA_INDEX(BIAS, b, oym, 0, 0);
        #else
            const uint bias_index = GET_DATA_INDEX(BIAS, b, ofm, 0, 0);
        #endif
    #elif BIAS_PER_OFM
        #if OUTPUT_3D
            const uint bias_index = oym;
        #else
            const uint bias_index = ofm;
        #endif
    #endif // BIAS_PER_OUTPUT
    ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(dotProd) + biases[bias_index];
#else
    ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(dotProd);
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE res = FUSED_OPS_RESULT;
    output[dst_index] = res;
#else
    output[dst_index] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(dequantized, ACTIVATION_PARAMS_TYPED));
#endif
}
