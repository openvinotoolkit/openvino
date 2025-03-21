// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/int4_utils.cl"

#ifdef INDEX_DIM
inline uint FUNC(get_positive_index)(OPTIONAL_SHAPE_INFO_ARG int in)
{
    if (in < 0)
        return in + INDEX_DIM;
    else
        return in;
}
#define INPUT_AXIS_INDEX (uint)FUNC_CALL(get_positive_index)(OPTIONAL_SHAPE_INFO_TENSOR indices[indices_idx])
#else
#define INPUT_AXIS_INDEX (uint)(indices[indices_idx])
#endif

#define GET_DICTIONARY_INDEX(idx_order) INPUT0_GET_INDEX(idx_order)
#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)

KERNEL(gather_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* dictionary,
    const __global INPUT1_TYPE* indices,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* decompression_scale,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* decompression_zp,
#endif
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
    const uint b = (uint)get_global_id(2) / OUTPUT_FEATURE_NUM;
    const uint f = (uint)get_global_id(2) % OUTPUT_FEATURE_NUM;
    #if OUTPUT_DIMS == 6
        #define ORDER b,f,w,z,y,x
        const uint w = (uint)get_global_id(1) / OUTPUT_SIZE_Z;
        const uint z = (uint)get_global_id(1) % OUTPUT_SIZE_Z;
        const uint y = (uint)get_global_id(0) / OUTPUT_SIZE_X;
        const uint x = (uint)get_global_id(0) % OUTPUT_SIZE_X;
    #elif OUTPUT_DIMS == 5
        #define ORDER b,f,z,y,x
        const uint z = (uint)get_global_id(1) / OUTPUT_SIZE_Y;
        const uint y = (uint)get_global_id(1) % OUTPUT_SIZE_Y;
        const uint x = (uint)get_global_id(0);
    #elif OUTPUT_DIMS == 4
        #define ORDER b,f,y,x
        const uint y = (uint)get_global_id(1);
        const uint x = (uint)get_global_id(0);
    #endif

    const uint indices_idx = GET_INDICES_INDEX(INDICES_INDEX_ORDER);
    const uint dictionary_idx = GET_DICTIONARY_INDEX(DICTIONARY_INDEX_ORDER);
    const uint output_idx = GET_INDEX(OUTPUT,,ORDER);

#if COMPRESSED_WEIGHTS
    OUTPUT_TYPE val = OUTPUT_VAL_ZERO;

    #if GATHER_AXIS_SHAPE_INFO_INDEX
        bool need_decompress = (INPUT_AXIS_INDEX >= 0 && INPUT_AXIS_INDEX < shape_info[GATHER_AXIS_SHAPE_INFO_INDEX]) ? true : false;
    #elif AXIS_DIM
        bool need_decompress = (INPUT_AXIS_INDEX >= 0 && INPUT_AXIS_INDEX < AXIS_DIM) ? true : false;
    #else
        bool need_decompress = true;
    #endif

    if (need_decompress) {
        #if DECOMPRESSION_ZP_TERM
            #if DECOMPRESSION_ZP_SCALAR
                OUTPUT_TYPE zp = DECOMPRESSION_ZP_VALUE;
            #else
                const uint zp_offset = dictionary_idx / DECOMPRESSION_ZP_GROUP_SIZE;
                OUTPUT_TYPE zp = TO_OUTPUT_TYPE(decompression_zp[zp_offset]);
            #endif
        #else
            OUTPUT_TYPE zp = OUTPUT_VAL_ZERO;
        #endif
        const uint decomp_offset = dictionary_idx / DECOMPRESSION_SCALE_GROUP_SIZE;
        DECOMPRESSION_SCALE_TYPE scale = decompression_scale[decomp_offset];

        #if COMPRESSED_WEIGHTS_INT8
            OUTPUT_TYPE val_compressed = dictionary[dictionary_idx];
            val = (val_compressed - zp) * scale;
        #elif COMPRESSED_WEIGHTS_INT4
            INPUT0_TYPE val_packed = dictionary[dictionary_idx / 2];
            MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) val_unpacked = UNPACK_INT4x2(OUTPUT_TYPE, *((INT4_PACKED_TYPE*)&val_packed));

            OUTPUT_TYPE val_compressed = ((OUTPUT_TYPE*)(&val_unpacked))[dictionary_idx % 2];
            val = (val_compressed - zp) * scale;
        #endif
    }
#else
    #if GATHER_AXIS_SHAPE_INFO_INDEX
        INPUT0_TYPE val = (INPUT_AXIS_INDEX >= 0 && INPUT_AXIS_INDEX < shape_info[GATHER_AXIS_SHAPE_INFO_INDEX]) ? dictionary[dictionary_idx] : 0;
    #elif AXIS_DIM
        INPUT0_TYPE val = (INPUT_AXIS_INDEX >= 0 && INPUT_AXIS_INDEX < AXIS_DIM) ? dictionary[dictionary_idx] : 0;
    #else
        INPUT0_TYPE val = dictionary[dictionary_idx];
    #endif
#endif

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT);
#else
    output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}

#undef GET_INDICES_INDEX
#undef GET_DICTIONARY_INDEX
#undef INPUT_AXIS_INDEX
