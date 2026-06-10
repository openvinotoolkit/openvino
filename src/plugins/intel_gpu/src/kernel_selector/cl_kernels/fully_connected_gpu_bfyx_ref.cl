// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/int4_utils.cl"

KERNEL(fc)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
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

if (get_global_id(0)==0 && get_global_id(1)==0 && get_global_id(2)==0) {
#if OUTPUT_3D
    printf("KERN %s: OUTPUT_3D\n", "fc");
#else
    printf("KERN %s: NON_3D\n", "fc");
#endif
}

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
            #if COMPRESSED_WEIGHTS
                #if DECOMPRESSION_ZP_TERM
                    #if DECOMPRESSION_ZP_SCALAR
                        ACCUMULATOR_TYPE zp = DECOMPRESSION_ZP_VALUE;
                    #else
                        const uint zp_offset = DECOMPRESSION_ZP_GET_INDEX_SAFE(oym, y / DECOMPRESSION_ZP_GROUP_SIZE, 0, 0);
                        ACCUMULATOR_TYPE zp = TO_ACCUMULATOR_TYPE(decompression_zp[zp_offset]);
                    #endif
                #else
                    ACCUMULATOR_TYPE zp = ACCUMULATOR_VAL_ZERO;
                #endif
                const uint decomp_offset = DECOMPRESSION_SCALE_GET_INDEX_SAFE(oym, y / DECOMPRESSION_SCALE_GROUP_SIZE, 0, 0);
                DECOMPRESSION_SCALE_TYPE scale = decompression_scale[decomp_offset];
            #endif

                const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, oym, y, 0, 0);
            #if COMPRESSED_WEIGHTS_INT8
                ACCUMULATOR_TYPE filter_compressed = TO_ACCUMULATOR_TYPE(weights[filter_idx]);
                ACCUMULATOR_TYPE filter_val = (filter_compressed - zp) * scale;
                dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * (ACCUMULATOR_TYPE)(filter_val);
            #elif COMPRESSED_WEIGHTS_INT4
                FILTER_TYPE filter_packed = weights[filter_idx / 2];
                MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 2) filter_unpacked = UNPACK_INT4x2(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&filter_packed));

                ACCUMULATOR_TYPE filter_compressed = ((ACCUMULATOR_TYPE*)(&filter_unpacked))[filter_idx % 2];
                ACCUMULATOR_TYPE filter_val = (filter_compressed - zp) * scale;
                dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * filter_val;
            #elif COMPRESSED_WEIGHTS_UINT2
                const __global uchar* weights_u8 = (const __global uchar*)weights;
                uchar filter_packed = weights_u8[filter_idx / 4];
                uint bit_offset = (filter_idx % 4) * 2;
                ACCUMULATOR_TYPE filter_compressed = (ACCUMULATOR_TYPE)((filter_packed >> bit_offset) & 0x3);
                 #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
                        const __global uchar* zp_u8 = (const __global uchar*)decompression_zp;
                        uchar zp_packed = zp_u8[zp_offset / 4];
                        uint zp_bit_off = (zp_offset % 4) * 2;
                        zp = (ACCUMULATOR_TYPE)((zp_packed >> zp_bit_off) & 0x3);
                    #endif

                ACCUMULATOR_TYPE filter_val = (filter_compressed - zp) * scale;
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
                #if COMPRESSED_WEIGHTS
                    #if DECOMPRESSION_ZP_TERM
                        #if DECOMPRESSION_ZP_SCALAR
                            ACCUMULATOR_TYPE zp = DECOMPRESSION_ZP_VALUE;
                        #else
                            #if COMPRESSED_WEIGHTS_UINT2
                                const uint num_zp_groups = INPUT0_FEATURE_NUM / DECOMPRESSION_ZP_GROUP_SIZE;
                                const uint zp_offset = ofm * num_zp_groups + (ifm / DECOMPRESSION_ZP_GROUP_SIZE);
                            #else
                                const uint zp_offset = DECOMPRESSION_ZP_GET_INDEX_SAFE(ofm, ifm / DECOMPRESSION_ZP_GROUP_SIZE, 0, 0);
                            #endif
                            ACCUMULATOR_TYPE zp = TO_ACCUMULATOR_TYPE(decompression_zp[zp_offset]);
                        #endif
                    #else
                        ACCUMULATOR_TYPE zp = ACCUMULATOR_VAL_ZERO;
                    #endif
                    const uint decomp_offset = DECOMPRESSION_SCALE_GET_INDEX_SAFE(ofm, ifm / DECOMPRESSION_SCALE_GROUP_SIZE, 0, 0);
                    DECOMPRESSION_SCALE_TYPE scale = decompression_scale[decomp_offset];
                #endif

                    const uint filter_idx = GET_FILTER_INDEX(FILTER, 0, ofm, ifm, y, x);
                #if COMPRESSED_WEIGHTS_INT8
                    FILTER_TYPE filter_compressed = weights[filter_idx];
                    ACCUMULATOR_TYPE filter_val = (TO_ACCUMULATOR_TYPE(filter_compressed) - zp) * scale;
                    dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * (ACCUMULATOR_TYPE)(filter_val);
                #elif COMPRESSED_WEIGHTS_INT4
                    FILTER_TYPE filter_packed = weights[filter_idx / 2];
                    MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, 2) filter_unpacked = UNPACK_INT4x2(ACCUMULATOR_TYPE, *((INT4_PACKED_TYPE*)&filter_packed));

                    ACCUMULATOR_TYPE filter_compressed = ((ACCUMULATOR_TYPE*)(&filter_unpacked))[filter_idx % 2];
                    ACCUMULATOR_TYPE filter_val = (filter_compressed - zp) * scale;
                    dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * filter_val;
                #elif COMPRESSED_WEIGHTS_UINT2
                   
                 
                    const __global uchar* weights_u8 = (const __global uchar*)weights;
                    uchar filter_packed = weights_u8[filter_idx / 4];
                    uint bit_offset = (filter_idx % 4) * 2;
                    ACCUMULATOR_TYPE filter_compressed = (ACCUMULATOR_TYPE)((filter_packed >> bit_offset) & 0x3);
                    #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
                        
                            const __global uchar* zp_u8 = (const __global uchar*)decompression_zp;
                            uchar zp_packed = zp_u8[zp_offset / 4];
                            uint zp_bit_off = (zp_offset % 4) * 2;
                            zp = (ACCUMULATOR_TYPE)((zp_packed >> zp_bit_off) & 0x3);
                       
                    #endif

                     if (get_global_id(0)==0 && get_global_id(1)==0 && ifm==0 && y==0 && x==0) {
                        #ifdef DECOMPRESSION_ZP_SCALAR
                            printf("DEFINES: ZP_GS=%d SC_GS=%d ZP_SCALAR=1 IFM=%d\n",
                                   DECOMPRESSION_ZP_GROUP_SIZE,
                                   DECOMPRESSION_SCALE_GROUP_SIZE,
                                   (int)INPUT0_FEATURE_NUM);
                        #else
                            printf("DEFINES: ZP_GS=%d SC_GS=%d ZP_SCALAR=0 IFM=%d\n",
                                   DECOMPRESSION_ZP_GROUP_SIZE,
                                   DECOMPRESSION_SCALE_GROUP_SIZE,
                                   (int)INPUT0_FEATURE_NUM);
                        #endif
                        printf("LAYOUT: ofm=%d ifm=%d filter_idx_at_0=%d\n",
                                OUTPUT_FEATURE_NUM, INPUT0_FEATURE_NUM,
                                GET_FILTER_INDEX(FILTER, 0, 0, 0, 0, 0));
                     }
                   
                    ACCUMULATOR_TYPE filter_val = (filter_compressed - zp) * scale;
                    dotProd += (ACCUMULATOR_TYPE)(input[input0_idx]) * filter_val;

             
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
    if (get_global_id(0) < 4 && b == 0) {
        printf("FC ofm=%u: dotProd=%f -> fused_res=%f\n",
               (uint)ofm, (float)dotProd, (float)res);
    }
    output[dst_index] = res;
#else
    if (get_global_id(0) < 4 && b == 0) {
        printf("FC ofm=%u: dotProd=%f (no fused)\n", (uint)ofm, (float)dotProd);
    }
    output[dst_index] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(dequantized, ACTIVATION_PARAMS_TYPED));
#endif
}
