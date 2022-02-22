// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#elif ELTWISE_NO_PITCH_SAME_DIMS
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _OFFSET) + idx_order
#else
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

KERNEL(eltwise)(
    INPUTS_DECLS
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{

#if OUTPUT_DIMS == 6 // 4D spatial
    #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
        uint data_idx = get_global_id(GWS_YX);
        const uint d1 = data_idx % OUTPUT_SIZE_X; // X
        data_idx = data_idx / OUTPUT_SIZE_X;

        const uint d2 = data_idx % OUTPUT_SIZE_Y; // Y
        data_idx = data_idx / OUTPUT_SIZE_Y;

        const uint d3 = data_idx % OUTPUT_SIZE_Z; // Z
        data_idx = data_idx / OUTPUT_SIZE_Z;

        const uint d4 = data_idx % OUTPUT_SIZE_W; // W

        const uint d5 = get_global_id(GWS_FEATURE);             // Feature
        const uint d6 = get_global_id(GWS_BATCH);               // Batch

        uint output_offset = OUTPUT_GET_INDEX(d6, d5, d4, d3, d2, d1);
    #elif ELTWISE_NO_PITCH_SAME_DIMS
        const uint d1 = get_global_id(0);
        uint output_offset = OUTPUT_OFFSET + d1;
    #else
        const uint d1 = get_global_id(0);
        const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
        const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1] % OUTPUT_SIZES[2];
        const uint d4 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2];
        const uint d5 = (uint)get_global_id(2) % OUTPUT_SIZES[4];
        const uint d6 = (uint)get_global_id(2) / OUTPUT_SIZES[4];

        uint output_offset = OUTPUT_GET_INDEX(d6, d5, d4, d3, d2, d1);
    #endif
#elif OUTPUT_DIMS == 5 // 3D spatial
    #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
        uint data_idx = get_global_id(GWS_YX);
        const uint d1 = data_idx % OUTPUT_SIZE_X; // X
        data_idx = data_idx / OUTPUT_SIZE_X;

        const uint d2 = data_idx % OUTPUT_SIZE_Y; // Y
        data_idx = data_idx / OUTPUT_SIZE_Y;

        const uint d3 = data_idx % OUTPUT_SIZE_Z; // Z

        const uint d4 = get_global_id(GWS_FEATURE);             // Feature
        const uint d5 = get_global_id(GWS_BATCH);               // Batch

        uint output_offset = OUTPUT_GET_INDEX(d5, d4, d3, d2, d1);
    #elif ELTWISE_NO_PITCH_SAME_DIMS
        const uint d1 = get_global_id(0);
        uint output_offset = OUTPUT_OFFSET + d1;
    #else
        const uint d1 = get_global_id(0);
        const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
        const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1];
        const uint d4 = (uint)get_global_id(2) % OUTPUT_SIZES[3];
        const uint d5 = (uint)get_global_id(2) / OUTPUT_SIZES[3];

        uint output_offset = OUTPUT_GET_INDEX(d5, d4, d3, d2, d1);
    #endif
#else // 2D spatial
    #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
        const uint d1 = (uint)get_global_id(GWS_YX) % OUTPUT_SIZE_X;  // X
        const uint d2 = (uint)get_global_id(GWS_YX) / OUTPUT_SIZE_X;  // Y
        const uint d3 = (uint)get_global_id(GWS_FEATURE);             // Feature
        const uint d4 = (uint)get_global_id(GWS_BATCH);               // Batch

        uint output_offset = GET_INDEX(OUTPUT,, OUTPUT_IDX_ORDER);
    #elif ELTWISE_NO_PITCH_SAME_DIMS
        const uint d1 = get_global_id(0);
        uint output_offset = OUTPUT_OFFSET + d1;
    #else
        const uint d1 = get_global_id(0);
        const uint d2 = get_global_id(1);
        const uint d3 = (uint)get_global_id(2) % OUTPUT_SIZES[2];
        const uint d4 = (uint)get_global_id(2) / OUTPUT_SIZES[2];

        uint output_offset = GET_INDEX(OUTPUT,, OUTPUT_IDX_ORDER);
    #endif
#endif

    ACCUMULATOR_TYPE res;

    DO_ELTWISE;

#if HAS_FUSED_OPS
    FUSED_OPS;
    OUTPUT_TYPE out = FUSED_OPS_RESULT;
#else
    #define out res
#endif

#if QUANTIZATION_TERM && !OUTPUT_IS_FP
    output[output_offset] = TO_OUTPUT_TYPE_SAT(ACTIVATION(out, ACTIVATION_PARAMS));
#else
    output[output_offset] = ACTIVATION_TYPED(out, ACTIVATION_PARAMS_TYPED);
#endif
}
