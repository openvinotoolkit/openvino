// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/data_types.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/imad.cl"

#define BYTES_PER_READ          (sizeof(int) * SIMD_SIZE)
#define BYTES_PER_READ8         (8 * BYTES_PER_READ)

#define AS_TYPE_N_(type, n, x)  as_##type##n(x)
#define AS_TYPE_N(type, n, x)   AS_TYPE_N_(type, n, x)
#define AS_INPUT0_TYPE_4(x)     AS_TYPE_N(INPUT0_TYPE, 4, x)

#if TILE_BATCH > 1

// TILE_OFM = 2, TILE_BATCH > 1, NO OUTPUT FEATURE LEFTOVERS
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL(fully_connected_gpu_imad)(
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
#if OUTPUT_3D
    const uint batch = (uint)get_global_id(1);
    const uint skip_f = (uint)get_global_id(2) * TILE_BATCH;
#else
    const uint batch = (uint)get_global_id(1) * TILE_BATCH;
    const uint skip_f = (uint)get_global_id(2);
#endif

    // Accumulators initialization
    MAKE_VECTOR_TYPE(int, TILE_OFM) dotProd[TILE_BATCH];
    MAKE_VECTOR_TYPE(uint, TILE_OFM) idx_w;
    __attribute__((opencl_unroll_hint))
    for (uint ob_idx = 0; ob_idx < TILE_BATCH; ob_idx++) {
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            dotProd[ob_idx][of_idx] = 0;
            idx_w[of_idx] = (((feature + of_idx * SIMD_SIZE) / SIMD_SIZE) * SIMD_SIZE) * IF_NUMBER;
        }
    }

    // Main calculation cycle by IFM
    __attribute__((opencl_unroll_hint(1)))
    for (uint idx_i = 0; idx_i < IF_NUMBER; idx_i += BYTES_PER_READ) {
        // Loading weights
        MAKE_VECTOR_TYPE(int, SIMD_SIZE) weights_data[TILE_OFM];
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
#if SIMD_SIZE > 8
            weights_data[of_idx].lo = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
            weights_data[of_idx].hi = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
#else
            weights_data[of_idx] = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
#endif
        }

        __attribute__((opencl_unroll_hint))
        for (uint ob_idx = 0; ob_idx < TILE_BATCH; ob_idx++) {
            // Loading inputs
        #if OUTPUT_3D
            __global INPUT0_TYPE* current_input = &input[GET_DATA_INDEX(INPUT0, batch, skip_f + ob_idx, 0, 0)];
        #else
            __global INPUT0_TYPE* current_input = &input[GET_DATA_INDEX(INPUT0, batch + ob_idx, skip_f, 0, 0)];
        #endif
            int input_data = as_int(intel_sub_group_block_read((const __global uint*)(current_input + idx_i)));
            __attribute__((opencl_unroll_hint))
            for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
                // Mad calculation
                __attribute__((opencl_unroll_hint))
                for (uint if_idx = 0; if_idx < SIMD_SIZE; if_idx++) {
                    // One IMAD macro produces upto 16 * 4 dot products (8 / 16 is SIMD and 4 is dp4a instruction)
                    dotProd[ob_idx][of_idx] = IMAD(dotProd[ob_idx][of_idx],
                                                   AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_data, if_idx)),
                                                   as_char4(weights_data[of_idx][if_idx]));
                }
            }
        }
    }

#if BIAS_TERM
    #if BIAS_PER_OUTPUT
        MAKE_VECTOR_TYPE(uint, TILE_OFM) bias_index[TILE_BATCH];
        __attribute__((opencl_unroll_hint))
        for (uint ob_idx = 0; ob_idx < TILE_BATCH; ob_idx++) {
            __attribute__((opencl_unroll_hint))
            for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            #if OUTPUT_3D
                bias_index[ob_idx][of_idx] = GET_DATA_INDEX(BIAS, batch, skip_f + ob_idx, feature + of_idx * SIMD_SIZE, 0);
            #else
                bias_index[ob_idx][of_idx] = GET_DATA_INDEX(BIAS, batch + ob_idx, feature + of_idx * SIMD_SIZE, 0, 0);
            #endif
            }
        }
    #elif BIAS_PER_OFM
        MAKE_VECTOR_TYPE(uint, TILE_OFM) bias_index;
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            bias_index[of_idx] = feature + of_idx * SIMD_SIZE;
        }
    #endif

    MAKE_VECTOR_TYPE(float, TILE_OFM) dequantized[TILE_BATCH];
    __attribute__((opencl_unroll_hint))
    for (uint ob_idx = 0; ob_idx < TILE_BATCH; ob_idx++) {
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if BIAS_PER_OUTPUT
            dequantized[ob_idx][of_idx] = (float)dotProd[ob_idx][of_idx] + biases[bias_index[ob_idx][of_idx]];
        #elif BIAS_PER_OFM
            dequantized[ob_idx][of_idx] = (float)dotProd[ob_idx][of_idx] + biases[bias_index[of_idx]];
        #endif
        }
    }
#else
    MAKE_VECTOR_TYPE(float, TILE_OFM) dequantized[TILE_BATCH];
    __attribute__((opencl_unroll_hint))
    for (uint ob_idx = 0; ob_idx < TILE_BATCH; ob_idx++) {
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            dequantized[ob_idx][of_idx] = (float)dotProd[ob_idx][of_idx];
        }
    }
#endif

#if HAS_FUSED_OPS
    __attribute__((opencl_unroll_hint))
    for (uint ob_idx = 0; ob_idx < TILE_BATCH; ob_idx++) {
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if OUTPUT_3D
            const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f + ob_idx, feature + of_idx * SIMD_SIZE, 0);
        #else
            const uint out_idx = OUTPUT_GET_INDEX(batch + ob_idx, feature + of_idx * SIMD_SIZE, 0, 0);
        #endif
            FUSED_OPS_BATCH_VEC;
            OUTPUT_TYPE res = FUSED_OPS_RESULT_BATCH_VEC;
            output[out_idx] = res;
        }
    }
#else
    __attribute__((opencl_unroll_hint))
    for (uint ob_idx = 0; ob_idx < TILE_BATCH; ob_idx++) {
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if OUTPUT_3D
            const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f + ob_idx, feature + of_idx * SIMD_SIZE, 0);
        #else
            const uint out_idx = OUTPUT_GET_INDEX(batch + ob_idx, feature + of_idx * SIMD_SIZE, 0, 0);
        #endif
            output[out_idx] = dequantized[ob_idx][of_idx];
        }
    }
#endif
}

#elif SLM_DIV_FACTOR > 1 // TILE_BATCH > 1

// TILE_OFM = 1, TILE_BATCH = 1, NO OUTPUT FEATURE LEFTOVERS, SLM SPLITTING TO IMPROVE OCCUPANCY
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL(fully_connected_gpu_imad)(
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
    const uint lid0 = (uint)get_local_id(0);
    const uint feature_per_wg = (uint)get_local_size(0) / SLM_DIV_FACTOR;
    const uint feature = (uint)get_group_id(0) * feature_per_wg + (uint)get_global_id(0) % feature_per_wg;
    const uint feature_block = lid0 / feature_per_wg;
    const uint batch = (uint)get_global_id(1);
    const uint skip_f = (uint)get_global_id(2);

    __local int partial_sum[WORK_GROUP_SIZE];

    // Accumulator initialization
    int dotProd = 0;
#if SIMD_SIZE > 8
    uint idx_w = ((feature / SIMD_SIZE) * SIMD_SIZE) * IF_NUMBER + feature_block * WORK_GROUPS_NUMBER * BYTES_PER_READ8 * 2;
#else
    uint idx_w = ((feature / SIMD_SIZE) * SIMD_SIZE) * IF_NUMBER + feature_block * WORK_GROUPS_NUMBER * BYTES_PER_READ8;
#endif

    // Main calculation cycle by IFM
    __attribute__((opencl_unroll_hint(1)))
    for (uint idx_i = feature_block * WORK_GROUPS_NUMBER * BYTES_PER_READ;
         idx_i < (feature_block + 1) * WORK_GROUPS_NUMBER * BYTES_PER_READ;
         idx_i += BYTES_PER_READ) {
        // Loading weights
        MAKE_VECTOR_TYPE(int, SIMD_SIZE) weights_data;
#if SIMD_SIZE > 8
        weights_data.lo = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w)));
        idx_w += BYTES_PER_READ8;
        weights_data.hi = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w)));
        idx_w += BYTES_PER_READ8;
#else
        weights_data = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w)));
        idx_w += BYTES_PER_READ8;
#endif

        // Loading input
        __global INPUT0_TYPE* current_input = &input[GET_DATA_INDEX(INPUT0, batch, skip_f, 0, 0)];
        int input_data = as_int(intel_sub_group_block_read((const __global uint*)(current_input + idx_i)));

        // Mad calculation
        __attribute__((opencl_unroll_hint))
        for (uint if_idx = 0; if_idx < SIMD_SIZE; if_idx++) {
            // One IMAD macro produces upto 16 * 4 dot products (8 / 16 is SIMD and 4 is dp4a instruction)
            dotProd = IMAD(dotProd,
                           AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_data, if_idx)),
                           as_char4(weights_data[if_idx]));
        }
    }

    partial_sum[lid0] = dotProd;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (feature_block == 0) {
        // Accumulate partial SLM sums into a one common sum
        __attribute__((opencl_unroll_hint))
        for (uint i = 1; i < SLM_DIV_FACTOR; i++)
            dotProd += partial_sum[lid0 % feature_per_wg + i * feature_per_wg];

    #if BIAS_TERM
        #if BIAS_PER_OUTPUT
            #if OUTPUT_3D
                const uint bias_index = GET_DATA_INDEX(BIAS, batch, skip_f, feature, 0);
            #else
                const uint bias_index = GET_DATA_INDEX(BIAS, batch, feature, 0, 0);
            #endif
        #elif BIAS_PER_OFM
            const uint bias_index = feature;
        #endif

        float dequantized = (float)dotProd + biases[bias_index];
    #else
        float dequantized = (float)dotProd;
    #endif

    #if OUTPUT_3D
        const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f, feature, 0);
    #else
        const uint out_idx = OUTPUT_GET_INDEX(batch, feature, 0, 0);
    #endif

    #if HAS_FUSED_OPS
        FUSED_OPS_SLM_SPLIT;
        OUTPUT_TYPE res = FUSED_OPS_RESULT_SLM_SPLIT;
        output[out_idx] = res;
    #else
        output[out_idx] = dequantized;
    #endif
    }
}

#else // TILE_BATCH > 1

// TILE_OFM = 1 OR 2, TILE_BATCH = 1, ANY OUTPUT FEATURE LEFTOVERS
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE)))
KERNEL(fully_connected_gpu_imad)(
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
    const uint batch = (uint)get_global_id(1);
    const uint skip_f = (uint)get_global_id(2);

    // Accumulators initialization
    int dotProd[TILE_OFM];
    uint idx_w[TILE_OFM];
    __attribute__((opencl_unroll_hint))
    for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        dotProd[of_idx] = 0;
        idx_w[of_idx] = (((feature + of_idx * SIMD_SIZE) / SIMD_SIZE) * SIMD_SIZE) * IF_NUMBER;
    }

    // Main calculation cycle by IFM
    __attribute__((opencl_unroll_hint(1)))
    for (uint idx_i = 0; idx_i < IF_NUMBER; idx_i += BYTES_PER_READ) {
        // Loading weights
        MAKE_VECTOR_TYPE(int, SIMD_SIZE) weights_data[TILE_OFM];
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
#if SIMD_SIZE > 8
            weights_data[of_idx].lo = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
            weights_data[of_idx].hi = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
#else
            weights_data[of_idx] = as_int8(intel_sub_group_block_read8((const __global uint*)(weights + idx_w[of_idx])));
            idx_w[of_idx] += BYTES_PER_READ8;
#endif
        }

        // Loading inputs
        __global INPUT0_TYPE* current_input = &input[GET_DATA_INDEX(INPUT0, batch, skip_f, 0, 0)];
        int input_data = as_int(intel_sub_group_block_read((const __global uint*)(current_input + idx_i)));

        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            // Mad calculation
            __attribute__((opencl_unroll_hint))
            for (uint if_idx = 0; if_idx < SIMD_SIZE; if_idx++) {
                // One IMAD macro produces upto 16 * 4 dot products (8 / 16 is SIMD and 4 is dp4a instruction)
                dotProd[of_idx] = IMAD(dotProd[of_idx],
                                       AS_INPUT0_TYPE_4(intel_sub_group_shuffle(input_data, if_idx)),
                                       as_char4(weights_data[of_idx][if_idx]));
            }
        }
    }

#if BIAS_TERM
    uint bias_index[TILE_OFM];
    #if BIAS_PER_OUTPUT
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        #if OUTPUT_3D
            bias_index[of_idx] = GET_DATA_INDEX(BIAS, batch, skip_f, feature + of_idx * SIMD_SIZE, 0);
        #else
            bias_index[of_idx] = GET_DATA_INDEX(BIAS, batch, feature + of_idx * SIMD_SIZE, 0, 0);
        #endif
        }
    #elif BIAS_PER_OFM
        __attribute__((opencl_unroll_hint))
        for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
            bias_index[of_idx] = feature + of_idx * SIMD_SIZE;
        }
    #endif

    float dequantized[TILE_OFM];
    __attribute__((opencl_unroll_hint))
    for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        dequantized[of_idx] = (float)dotProd[of_idx] + biases[bias_index[of_idx]];
    }
#else
    float dequantized[TILE_OFM];
    __attribute__((opencl_unroll_hint))
    for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
        dequantized[of_idx] = (float)dotProd[of_idx];
    }
#endif

#if HAS_FUSED_OPS
    __attribute__((opencl_unroll_hint))
    for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
    #if OUTPUT_3D
        const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f, feature + of_idx * SIMD_SIZE, 0);
    #else
        const uint out_idx = OUTPUT_GET_INDEX(batch, feature + of_idx * SIMD_SIZE, 0, 0);
    #endif
        FUSED_OPS_BATCH_SCALAR;
        OUTPUT_TYPE res = FUSED_OPS_RESULT_BATCH_SCALAR;
    #if HAS_OUTPUT_LEFTOVERS
        if (feature < OF_NUMBER)
    #endif
        {
            output[out_idx] = res;
        }
    }
#else
    __attribute__((opencl_unroll_hint))
    for (uint of_idx = 0; of_idx < TILE_OFM; of_idx++) {
    #if HAS_OUTPUT_LEFTOVERS
        if (feature < OF_NUMBER)
    #endif
        {
        #if OUTPUT_3D
            const uint out_idx = OUTPUT_GET_INDEX(batch, skip_f, feature + of_idx * SIMD_SIZE, 0);
        #else
            const uint out_idx = OUTPUT_GET_INDEX(batch, feature + of_idx * SIMD_SIZE, 0, 0);
        #endif
            output[out_idx] = dequantized[of_idx];
        }
    }
#endif // HAS_FUSED_OPS
}

#endif // TILE_BATCH > 1

#undef BYTES_PER_READ8
#undef BYTES_PER_READ
#undef AS_INPUT0_TYPE_4
#undef AS_TYPE_N
#undef AS_TYPE_N_
