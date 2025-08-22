// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX_SAFE)(idx_order)
#elif ELTWISE_NO_PITCH_SAME_DIMS
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _OFFSET) + idx_order
#else
    #define GET_INDEX(prefix, num, idx_order) CAT(CAT(prefix, num), _GET_INDEX)(idx_order)
#endif

KERNEL(eltwise)(
    OPTIONAL_SHAPE_INFO_ARG
    INPUTS_DECLS
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
#if IS_DYNAMIC_CROP
    , int runtime_offset
#endif
)
{

#if OUTPUT_DIMS == 8 // 6D spatial
    #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
        uint data_idx = get_global_id(GWS_YX);
        const uint d1 = data_idx % OUTPUT_SIZE_X; // X
        data_idx = data_idx / OUTPUT_SIZE_X;

        const uint d2 = data_idx % OUTPUT_SIZE_Y; // Y
        data_idx = data_idx / OUTPUT_SIZE_Y;

        const uint d3 = data_idx % OUTPUT_SIZE_Z; // Z
        data_idx = data_idx / OUTPUT_SIZE_Z;

        const uint d4 = data_idx % OUTPUT_SIZE_W; // W
        data_idx = data_idx / OUTPUT_SIZE_W;

        const uint d5 = data_idx % OUTPUT_SIZE_U; // U
        data_idx = data_idx / OUTPUT_SIZE_U;

        const uint d6 = data_idx % OUTPUT_SIZE_V; // V

        const uint d7 = get_global_id(GWS_FEATURE);             // Feature
        const uint d8 = get_global_id(GWS_BATCH);               // Batch

        uint output_offset = OUTPUT_GET_INDEX(d8, d7, d6, d5, d4, d3, d2, d1);
    #elif ELTWISE_NO_PITCH_SAME_DIMS
        const uint d1 = get_global_id(0);
        uint output_offset = OUTPUT_OFFSET + d1;
    #else
        const uint d1 = get_global_id(0);
        const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
        const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1] % OUTPUT_SIZES[2];
        const uint d4 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] % OUTPUT_SIZES[3];
        const uint d5 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] / OUTPUT_SIZES[3] % OUTPUT_SIZES[4];
        const uint d6 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] / OUTPUT_SIZES[3] / OUTPUT_SIZES[4];
        const uint d7 = (uint)get_global_id(2) % OUTPUT_SIZES[6];
        const uint d8 = (uint)get_global_id(2) / OUTPUT_SIZES[6];

        uint output_offset = OUTPUT_GET_INDEX(d8, d7, d6, d5, d4, d3, d2, d1);
    #endif
#elif OUTPUT_DIMS == 7 // 5D spatial
    #if ELTWISE_LAYOUT_BASED || QUANTIZATION_TERM || ELTWISE_BROADCAST
        uint data_idx = get_global_id(GWS_YX);
        const uint d1 = data_idx % OUTPUT_SIZE_X; // X
        data_idx = data_idx / OUTPUT_SIZE_X;

        const uint d2 = data_idx % OUTPUT_SIZE_Y; // Y
        data_idx = data_idx / OUTPUT_SIZE_Y;

        const uint d3 = data_idx % OUTPUT_SIZE_Z; // Z
        data_idx = data_idx / OUTPUT_SIZE_Z;

        const uint d4 = data_idx % OUTPUT_SIZE_W; // W
        data_idx = data_idx / OUTPUT_SIZE_W;

        const uint d5 = data_idx % OUTPUT_SIZE_U; // U

        const uint d6 = get_global_id(GWS_FEATURE);             // Feature
        const uint d7 = get_global_id(GWS_BATCH);               // Batch

        uint output_offset = OUTPUT_GET_INDEX(d7, d6, d5, d4, d3, d2, d1);
    #elif ELTWISE_NO_PITCH_SAME_DIMS
        const uint d1 = get_global_id(0);
        uint output_offset = OUTPUT_OFFSET + d1;
    #else
        const uint d1 = get_global_id(0);
        const uint d2 = (uint)get_global_id(1) % OUTPUT_SIZES[1];
        const uint d3 = (uint)get_global_id(1) / OUTPUT_SIZES[1] % OUTPUT_SIZES[2];
        const uint d4 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] % OUTPUT_SIZES[3];
        const uint d5 = (uint)get_global_id(1) / OUTPUT_SIZES[1] / OUTPUT_SIZES[2] / OUTPUT_SIZES[3];
        const uint d6 = (uint)get_global_id(2) % OUTPUT_SIZES[5];
        const uint d7 = (uint)get_global_id(2) / OUTPUT_SIZES[5];

        uint output_offset = OUTPUT_GET_INDEX(d7, d6, d5, d4, d3, d2, d1);
    #endif
#elif OUTPUT_DIMS == 6 // 4D spatial
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

        // zero-padding the blocked format padded memory area since it might be used as input of onednn concatenation
        if(d4 + d3 + d2 + d1 == 0) {
            const uint b_size = OUTPUT_SIZES[3], f_size = OUTPUT_SIZES[2], y_size = OUTPUT_SIZES[1], x_size = OUTPUT_SIZES[0];

            #if BATCH_BLOCK_SIZE && FEATURE_BLOCK_SIZE
                const uint padded_fs = (f_size + FEATURE_BLOCK_SIZE -1) / FEATURE_BLOCK_SIZE;
                const uint padded_bs = (b_size + BATCH_BLOCK_SIZE -1) / BATCH_BLOCK_SIZE;
                const uint z_size = 1;

                const uint bsv_pitch = FEATURE_BLOCK_SIZE;
                const uint x_pitch = bsv_pitch * BATCH_BLOCK_SIZE;
                const uint y_pitch = x_pitch * x_size;
                const uint z_pitch = y_pitch * y_size;
                const uint fs_pitch = z_pitch * z_size;
                const uint bs_pitch = fs_pitch * padded_fs;

                uint b = 0;
                uint f = 0;
                uint offset = 0;
                for (uint bs = 0; bs < padded_bs; ++bs) {
                    for (uint fs = 0; fs < padded_fs; ++fs) {
                        for (uint z = 0; z < z_size; ++z) {
                            for (uint y = 0; y < y_size; ++y) {
                                for (uint x = 0; x < x_size; ++x) {
                                    for (uint bsv = 0; bsv < BATCH_BLOCK_SIZE; ++bsv) {
                                        for (uint fsv = 0; fsv < FEATURE_BLOCK_SIZE; ++fsv) {
                                            b = bs * BATCH_BLOCK_SIZE + bsv;
                                            f = fs * FEATURE_BLOCK_SIZE + fsv;
                                            if(b >= b_size || f >= f_size) {
                                                offset = bs * bs_pitch + fs * fs_pitch + z * z_pitch +
                                                         y * y_pitch + x * x_pitch + bsv * bsv_pitch + fsv;
                                                output[offset] = 0;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            #elif FEATURE_BLOCK_SIZE
                const uint padded_fs = (f_size + FEATURE_BLOCK_SIZE -1) / FEATURE_BLOCK_SIZE;

                const uint x_pitch = FEATURE_BLOCK_SIZE;
                const uint y_pitch = x_pitch * x_size;
                const uint fs_pitch = y_pitch * y_size;
                const uint b_pitch = fs_pitch * padded_fs;

                uint f = 0;
                uint offset = 0;
                for (uint b = 0; b < b_size; ++b) {
                    for (uint fs = padded_fs - 1; fs < padded_fs; ++fs) {
                        for (uint y = 0; y < y_size; ++y) {
                            for (uint x = 0; x < x_size; ++x) {
                                for (uint fsv = 0; fsv < FEATURE_BLOCK_SIZE; ++fsv) {
                                    f = fs * FEATURE_BLOCK_SIZE + fsv;
                                    if(f >= f_size) {
                                        offset = b * b_pitch + fs * fs_pitch + y * y_pitch + x * x_pitch + fsv;
                                        output[offset] = 0;
                                    }
                                }
                            }
                        }
                    }
                }
            #endif
        }
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
    output[output_offset] = TO_OUTPUT_TYPE(ACTIVATION_TYPED(out, ACTIVATION_PARAMS_TYPED));
#endif
}
