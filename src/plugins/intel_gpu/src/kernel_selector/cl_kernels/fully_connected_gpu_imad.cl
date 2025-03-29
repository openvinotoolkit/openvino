// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/imad.cl"

#define BYTES_PER_READ          (PACK_SIZE * SIMD_SIZE)
#define BYTES_PER_READ8         (8 * BYTES_PER_READ)

#define AS_TYPE_N_(type, n, x)  as_##type##n(x)
#define AS_TYPE_N(type, n, x)   AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x)     AS_TYPE_N(INPUT0_TYPE, 4, x)

REQD_SUB_GROUP_SIZE(SIMD_SIZE)
KERNEL(fully_connected_gpu_imad)(
    OPTIONAL_SHAPE_INFO_ARG
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
    const uint feature = (uint)get_group_id(0) * SIMD_SIZE * TILE_OFM + (uint)get_global_id(0) % SIMD_SIZE;
#if HAS_OFM_LEFTOVERS
    // Sign that last ofm block is in processing
    const bool last_feature_block = (uint)get_group_id(0) == ((uint)get_num_groups(0) - 1);
#endif
#if HAS_OFM_LEFTOVERS || HAS_IFM_LEFTOVERS
    const uint sglid = get_sub_group_local_id();
#endif
#if IS_DYNAMIC
    // In dynamic kernel, TILE_BATCH is set to the initial tile batch size for stack arrays such as dotProd
    // and tile_batch is calculated as an adjusted value from tile_batch_max_size by given global work size
#if OUTPUT_3D
    const uint tile_batch = OUTPUT_FEATURE_NUM / (uint)get_global_size(2);
#else
    const uint tile_batch = OUTPUT_BATCH_NUM / (uint)get_global_size(1);
#endif
#else
    const uint tile_batch = TILE_BATCH;
#endif

#if OUTPUT_3D
    const uint batch = (uint)get_global_id(1);
    const uint skip_f = (uint)get_global_id(2) * tile_batch;
#else
    const uint batch = (uint)get_global_id(1) * tile_batch;
    const uint skip_f = (uint)get_global_id(2);
#endif

    // Accumulators initialization
    MAKE_VECTOR_TYPE(int, TILE_OFM) dotProd[TILE_BATCH];
    MAKE_VECTOR_TYPE(uint, TILE_OFM) idx_w;
#if IS_DYNAMIC
    for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#else
    unroll_for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#endif
        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            dotProd[ob_idx][of_idx] = 0;
        #if !HAS_IFM_LEFTOVERS
            idx_w[of_idx] = (((feature + of_idx * SIMD_SIZE) / SIMD_SIZE) * SIMD_SIZE) * IF_NUMBER;
        #else
            idx_w[of_idx] = (((feature + of_idx * SIMD_SIZE) / SIMD_SIZE) * SIMD_SIZE) * (((IF_NUMBER + PACK_SIZE - 1) / PACK_SIZE) * PACK_SIZE);
        #endif
        }
    }

    // Main calculation cycle by IFM
    __attribute__((opencl_unroll_hint(1)))
#if !HAS_IFM_LEFTOVERS
    for (uint idx_i = 0; idx_i < IF_NUMBER; idx_i += BYTES_PER_READ) {
#else
    for (uint idx_i = 0; idx_i < IF_NUMBER - IF_NUMBER % BYTES_PER_READ; idx_i += BYTES_PER_READ) {
#endif
        // Loading weights
        MAKE_VECTOR_TYPE(int, SIMD_SIZE) weights_data[TILE_OFM];
        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
    #if !HAS_OFM_LEFTOVERS
        #if SIMD_SIZE > 8
            weights_data[of_idx].lo = as_int8(_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
            weights_data[of_idx].hi = as_int8(_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
        #else
            weights_data[of_idx] = as_int8(_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
        #endif
    #else // !HAS_OFM_LEFTOVERS
            if (!last_feature_block) {
            #if SIMD_SIZE > 8
                weights_data[of_idx].lo = as_int8(_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
                idx_w[of_idx] += BYTES_PER_READ8;
                weights_data[of_idx].hi = as_int8(_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
                idx_w[of_idx] += BYTES_PER_READ8;
            #else
                weights_data[of_idx] = as_int8(_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
                idx_w[of_idx] += BYTES_PER_READ8;
            #endif
            } else {
                weights_data[of_idx] = 0;
                unroll_for (uint if_idx = 0; if_idx < SIMD_SIZE; if_idx++) {
                    if (feature + of_idx * SIMD_SIZE < OF_NUMBER) {
                        __global int* wei_ptr = (__global int*)weights;
                        weights_data[of_idx][if_idx] = wei_ptr[idx_w[of_idx] / PACK_SIZE + if_idx * BYTES_PER_READ / PACK_SIZE + sglid];
                    }
                }
                idx_w[of_idx] += BYTES_PER_READ * SIMD_SIZE;
            }
    #endif // HAS_OFM_LEFTOVERS
        }

    #if IS_DYNAMIC
        for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
    #else
        unroll_for(uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
    #endif
            // Loading inputs
        #if OUTPUT_3D
            __global INPUT0_TYPE* current_input = &input[INPUT0_GET_INDEX(batch, skip_f + ob_idx, 0, 0)];
        #else
            __global INPUT0_TYPE* current_input = &input[INPUT0_GET_INDEX(batch + ob_idx, skip_f, 0, 0)];
        #endif

        #if !HAS_IFM_LEFTOVERS
            int input_data = as_int(_sub_group_block_read((const __global uint*)(current_input + idx_i)));
        #else
            MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) temp_input = { current_input[idx_i + sglid * PACK_SIZE],
                                                                    current_input[idx_i + sglid * PACK_SIZE + 1],
                                                                    current_input[idx_i + sglid * PACK_SIZE + 2],
                                                                    current_input[idx_i + sglid * PACK_SIZE + 3] };
            int input_data = as_int(temp_input);
        #endif

            unroll_for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
                // Mad calculation
                unroll_for (uint if_idx = 0; if_idx < SIMD_SIZE; if_idx++) {
                    // One IMAD macro produces upto 16 * 4 dot products (8 / 16 is SIMD and 4 is dp4a instruction)
                    dotProd[ob_idx][of_idx] = IMAD(dotProd[ob_idx][of_idx],
                                                   AS_INPUT0_TYPE_4(_sub_group_shuffle(input_data, if_idx)),
                                                   as_char4(weights_data[of_idx][if_idx]));
                }
            }
        }
    }


    // Main calculation cycle by IFM (leftovers)
#if HAS_IFM_LEFTOVERS
    // Loading weights
    MAKE_VECTOR_TYPE(int, SIMD_SIZE) weights_data[TILE_OFM];
    unroll_for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        weights_data[of_idx] = (MAKE_VECTOR_TYPE(int, SIMD_SIZE))0;
        unroll_for(uint if_idx = 0; if_idx < SIMD_SIZE; if_idx++) {
            char4 temp_weights = (char4)0;
        #if !HAS_OFM_LEFTOVERS
            if (if_idx * PACK_SIZE < IF_NUMBER % BYTES_PER_READ) {
        #else
            if (feature + of_idx * SIMD_SIZE < OF_NUMBER && if_idx * PACK_SIZE < IF_NUMBER % BYTES_PER_READ) {
        #endif
                uint wei_offset = idx_w[of_idx] + if_idx * BYTES_PER_READ + sglid * PACK_SIZE;

                if ((if_idx + 1) * PACK_SIZE <= IF_NUMBER % BYTES_PER_READ) {
                    temp_weights.s0 = weights[wei_offset];
                    temp_weights.s1 = weights[wei_offset + 1];
                    temp_weights.s2 = weights[wei_offset + 2];
                    temp_weights.s3 = weights[wei_offset + 3];
                } else {
                    temp_weights.s0 = weights[wei_offset];
                #if IF_NUMBER % PACK_SIZE > 1
                    temp_weights.s1 = weights[wei_offset + 1];
                #endif
                #if IF_NUMBER % PACK_SIZE > 2
                    temp_weights.s2 = weights[wei_offset + 2];
                #endif
                #if IF_NUMBER % PACK_SIZE > 3
                    temp_weights.s3 = weights[wei_offset + 3];
                #endif
                }

                weights_data[of_idx][if_idx] = as_int(temp_weights);
            }
        }
    }

#if IS_DYNAMIC
    for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#else
    unroll_for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#endif
        // Loading inputs
    #if OUTPUT_3D
        __global INPUT0_TYPE* current_input = &input[INPUT0_GET_INDEX(batch, skip_f + ob_idx, 0, 0)];
    #else
        __global INPUT0_TYPE* current_input = &input[INPUT0_GET_INDEX(batch + ob_idx, skip_f, 0, 0)];
    #endif
        int input_data = 0;
        MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE) temp_input = (MAKE_VECTOR_TYPE(INPUT0_TYPE, PACK_SIZE))0;

        if (sglid * PACK_SIZE < IF_NUMBER % BYTES_PER_READ) {
            uint in_offset = IF_NUMBER - IF_NUMBER % BYTES_PER_READ + sglid * PACK_SIZE;

            if ((sglid + 1) * PACK_SIZE <= IF_NUMBER % BYTES_PER_READ) {
                temp_input.s0 = current_input[in_offset];
                temp_input.s1 = current_input[in_offset + 1];
                temp_input.s2 = current_input[in_offset + 2];
                temp_input.s3 = current_input[in_offset + 3];
            } else {
                temp_input.s0 = current_input[in_offset];
            #if IF_NUMBER % PACK_SIZE > 1
                temp_input.s1 = current_input[in_offset + 1];
            #endif
            #if IF_NUMBER % PACK_SIZE > 2
                temp_input.s2 = current_input[in_offset + 2];
            #endif
            #if IF_NUMBER % PACK_SIZE > 3
                temp_input.s3 = current_input[in_offset + 3];
            #endif
            }

            input_data = as_int(temp_input);
        }

        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            // Mad calculation
            unroll_for (uint if_idx = 0; if_idx < SIMD_SIZE; if_idx++) {
                // One IMAD macro produces upto 16 * 4 dot products (8 / 16 is SIMD and 4 is dp4a instruction)
                dotProd[ob_idx][of_idx] = IMAD(dotProd[ob_idx][of_idx],
                                               AS_INPUT0_TYPE_4(_sub_group_shuffle(input_data, if_idx)),
                                               as_char4(weights_data[of_idx][if_idx]));
            }
        }
    }
#endif // HAS_IFM_LEFTOVERS

#if BIAS_TERM
    #if BIAS_PER_OUTPUT
        MAKE_VECTOR_TYPE(uint, TILE_OFM) bias_index[TILE_BATCH];
    #if IS_DYNAMIC
        for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
    #else
        unroll_for(uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
    #endif
            unroll_for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            #if OUTPUT_3D
                bias_index[ob_idx][of_idx] = GET_DATA_INDEX(BIAS, batch, skip_f + ob_idx, feature + of_idx * SIMD_SIZE, 0);
            #else
                bias_index[ob_idx][of_idx] = GET_DATA_INDEX(BIAS, batch + ob_idx, feature + of_idx * SIMD_SIZE, 0, 0);
            #endif
            }
        }
    #elif BIAS_PER_OFM
        MAKE_VECTOR_TYPE(uint, TILE_OFM) bias_index;
        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            bias_index[of_idx] = feature + of_idx * SIMD_SIZE;
        }
    #endif

    MAKE_VECTOR_TYPE(float, TILE_OFM) dequantized[TILE_BATCH];
#if IS_DYNAMIC
    for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#else
    unroll_for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#endif
        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if HAS_OFM_LEFTOVERS
            if (feature + of_idx * SIMD_SIZE < OF_NUMBER)
        #endif // HAS_OFM_LEFTOVERS
            #if BIAS_PER_OUTPUT
                dequantized[ob_idx][of_idx] = (float)dotProd[ob_idx][of_idx] + biases[bias_index[ob_idx][of_idx]];
            #elif BIAS_PER_OFM
                dequantized[ob_idx][of_idx] = (float)dotProd[ob_idx][of_idx] + biases[bias_index[of_idx]];
            #endif // BIAS_PER_OFM
        }
    }
#else
    MAKE_VECTOR_TYPE(float, TILE_OFM) dequantized[TILE_BATCH];
#if IS_DYNAMIC
    for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#else
    unroll_for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#endif
        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if HAS_OFM_LEFTOVERS
            if (feature + of_idx * SIMD_SIZE < OF_NUMBER)
        #endif
                dequantized[ob_idx][of_idx] = (float)dotProd[ob_idx][of_idx];
        }
    }
#endif

#if HAS_FUSED_OPS
#if IS_DYNAMIC
    for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#else
    unroll_for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#endif
        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if HAS_OFM_LEFTOVERS
            if (feature + of_idx * SIMD_SIZE < OF_NUMBER) {
        #endif
            #if OUTPUT_3D
                const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f + ob_idx, feature + of_idx * SIMD_SIZE, 0);
            #else
                const uint out_idx = OUTPUT_GET_INDEX(batch + ob_idx, feature + of_idx * SIMD_SIZE, 0, 0);
            #endif
                FUSED_OPS_BATCH_VEC;
                OUTPUT_TYPE res = FUSED_OPS_RESULT_BATCH_VEC;
                output[out_idx] = res;
        #if HAS_OFM_LEFTOVERS
            }
        #endif
        }
    }
#else
#if IS_DYNAMIC
    for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#else
    unroll_for (uint ob_idx = 0; ob_idx < tile_batch; ob_idx++) {
#endif
        unroll_for(uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if HAS_OFM_LEFTOVERS
            if (feature + of_idx * SIMD_SIZE < OF_NUMBER) {
        #endif
            #if OUTPUT_3D
                const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f + ob_idx, feature + of_idx * SIMD_SIZE, 0);
            #else
                const uint out_idx = OUTPUT_GET_INDEX(batch + ob_idx, feature + of_idx * SIMD_SIZE, 0, 0);
            #endif
                output[out_idx] = dequantized[ob_idx][of_idx];
        #if HAS_OFM_LEFTOVERS
            }
        #endif
        }
    }
#endif // HAS_FUSED_OPS
}

#undef BYTES_PER_READ8
#undef BYTES_PER_READ
#undef AS_INPUT0_TYPE_4
#undef AS_TYPE_N
#undef AS_TYPE_N_
