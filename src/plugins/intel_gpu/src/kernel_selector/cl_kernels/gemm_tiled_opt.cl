// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#if INPUT0_TYPE_SIZE == 4
#define BLOCK_SHUFFLE               _sub_group_shuffle
#else // INPUT0_TYPE_SIZE == 4
#define BLOCK_SHUFFLE(data, sg_lid) as_half16(_sub_group_shuffle(as_short16(data), sg_lid))
#endif // INPUT0_TYPE_SIZE == 4

#if TILE_K > SIMD_WIDTH
    #define BLOCK_READ_A(ptr, offset) BLOCK_READN(INPUT0_TYPE, A_VEC_SIZE, ptr, offset)
#else // TILE_K > SIMD_WIDTH
    #define BLOCK_READ_A(ptr, offset) BLOCK_READN(INPUT0_TYPE, 1, ptr, offset)
#endif // TILE_K > SIMD_WIDTH

#if TILE_N > SIMD_WIDTH
    #define BLOCK_READ_B(ptr, offset) BLOCK_READN(INPUT1_TYPE, B_VEC_SIZE, ptr, offset)
    #define BLOCK_WRITE_C(ptr, offset, data) BLOCK_WRITEN(OUTPUT_TYPE, B_VEC_SIZE, ptr, offset, data)
#else // TILE_N > SIMD_WIDTH
    #define BLOCK_READ_B(ptr, offset) BLOCK_READN(INPUT1_TYPE, 1, ptr, offset)
    #define BLOCK_WRITE_C(ptr, offset, data) BLOCK_WRITEN(OUTPUT_TYPE, 1, ptr, offset, data)
#endif // TILE_N > SIMD_WIDTH

inline uint FUNC(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, y, x);
#else
#   error gemm_tiled_opt.cl : Unsupported input 0 format
#endif
}

inline uint FUNC(get_input0_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_input0_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT0_DIMS_ORDER);
}

inline uint FUNC(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, y, x);
#else
#   error gemm_tiled_opt.cl : Unsupported input 1 format
#endif
}

inline uint FUNC(get_input1_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_batch_offset)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_tiled_opt.cl : Unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // INPUT2_TYPE

#define VLOAD CAT(vload, SIMD_WIDTH)

// Optimized gemm kernel for fp16/fp32 inputs
REQD_SUB_GROUP_SIZE(SIMD_WIDTH)
__attribute__((reqd_work_group_size(SIMD_WIDTH, 1, 1)))
KERNEL(gemm_tiled_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* input2,
#endif // INPUT2_TYPE
#if BEAM_TABLE_TERM
    const __global BEAM_TABLE_TYPE* beam_table,
#endif
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif // HAS_FUSED_OPS_DECLS
    )
{
    const uint tile_n_num = (uint)get_group_id(0);
    const uint tile_m_num = (uint)get_group_id(1);
    const uint tile_m_size = (uint)get_global_size(1);
    const uint tile_m_offset = tile_m_num * TILE_M;
    const uint tile_n_offset = tile_n_num * TILE_N;
    uint batch_number = (uint)get_global_id(2);
    const uint sglid = (uint)get_sub_group_local_id();

    // Setting x and y for fusings indexing
    // TODO: investigate how we can use only TILE_N_NOT_DIVISIBLE here for getting stable results in fusings
#if IS_DYNAMIC
    const uint x = (uint)get_global_id(0);
#else // IS_DYNAMIC
#if TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
    const uint x = (uint)get_global_id(0);
#else // TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
    const uint x = tile_n_num * SIMD_WIDTH * B_VEC_SIZE;
#endif // TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
#endif // IS_DYNAMIC
    uint y = tile_m_offset;

    const uint tile_m_iterations = TILE_M_NOT_DIVISIBLE ? (tile_m_num == (tile_m_size - 1) ? TILE_M_LEFTOVER : TILE_M) : TILE_M;
    const uint z = batch_number % TR_OUTPUT_SIZE_Z;
    batch_number /= TR_OUTPUT_SIZE_Z;
    const uint w = batch_number % TR_OUTPUT_SIZE_W;
    batch_number /= TR_OUTPUT_SIZE_W;
    const uint f = batch_number % TR_OUTPUT_FEATURE_NUM;
    batch_number /= TR_OUTPUT_FEATURE_NUM;
    const uint b = batch_number % TR_OUTPUT_BATCH_NUM;

    // Batch offsets
    const uint batch_offset_input0 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, 0);
    const uint batch_offset_input1 = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, 0, tile_n_offset);
#ifdef INPUT2_TYPE
    const uint batch_offset_input2 = FUNC_CALL(get_input2_batch_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z);
#endif // INPUT2_TYPE
    uint write_id = 0;
    const uint batch_offset_output = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR TR_B, TR_F, TR_W, TR_Z, TR_Y, TR_X);
    write_id = 1;
    const uint batch_offset_output_diff = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR TR_B, TR_F, TR_W, TR_Z, TR_Y, TR_X) - batch_offset_output;

    // Start pointers offsets
#if TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST
    const __global INPUT0_TYPE* a_ptr = input0 + batch_offset_input0;
    #if HAS_DYNAMIC_K_PADDING || INPUT0_HAS_PADDING
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y+1), 0) - batch_offset_input0;
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, (TILE_K)) - batch_offset_input0;
    #else
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 1, 0);
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, (TILE_K));
    #endif
#elif TRANSPOSE_INPUT0 == TRANSPOSE_Y_LAST
    const __global INPUT0_TYPE* a_ptr = input0 + batch_offset_input0;
    #if HAS_DYNAMIC_K_PADDING || INPUT0_HAS_PADDING
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, 1) - batch_offset_input0;
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, (TILE_K)) - batch_offset_input0;
    #else
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, 1);
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, (TILE_K));
    #endif
#endif // TRANSPOSE_INPUT0
#if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
    const __global INPUT1_TYPE* b_ptr = input1 + batch_offset_input1;
    #if HAS_DYNAMIC_K_PADDING || INPUT1_HAS_PADDING
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, 1, tile_n_offset) - batch_offset_input1;
    #else
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 1, 0);
    #endif
#elif TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
    const __global INPUT1_TYPE* b_ptr = input1 + batch_offset_input1;
    #if HAS_DYNAMIC_K_PADDING || INPUT1_HAS_PADDING
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, 0, (tile_n_offset + 1)) - batch_offset_input1;
        const uint input1_offset1 = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (TILE_K), tile_n_offset) - batch_offset_input1;
    #else
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, 1);
        const uint input1_offset1 = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, (TILE_K), 0);
    #endif
    const uint input1_fetch_size = ((N - tile_n_offset) < TILE_K) ? (N - tile_n_offset) : TILE_K;
#endif // TRANSPOSE_INPUT1
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* c_ptr = input2 + batch_offset_input2 + tile_m_offset * N + tile_n_offset;
#endif // INPUT2_TYPE
    __global OUTPUT_TYPE* d_ptr = output + batch_offset_output;

    const uint b_raw_global_id = tile_n_offset + sglid;

#if TRANSPOSE_INPUT0 != TRANSPOSE_X_LAST
    MAKE_VECTOR_TYPE(INPUT0_TYPE, SIMD_WIDTH) a_tile;
#endif // TRANSPOSE_INPUT0 != TRANSPOSE_X_LAST
#if TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
    B_FLOATN b_tile[TILE_K];
#else // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
    MAKE_VECTOR_TYPE(INPUT1_TYPE, SIMD_WIDTH) b_tile;
#endif // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
    B_FLOATN c_tile[TILE_M];

    unroll_for (uint i = 0; i < TILE_M; i++) {
        c_tile[i] = (B_FLOATN)(ACCUMULATOR_VAL_ZERO);
    }

    // Full tile calculation
    for (uint k = 0; k < K_FULL_ITERATIONS; k++) {

        // Loading B tile
        unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
#if IS_DYNAMIC
#if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
#if HAS_DYNAMIC_N_PADDING || INPUT1_HAS_PADDING
            b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
#else
            b_tile[b_load_id] = TILE_N_NOT_DIVISIBLE ? (b_raw_global_id > N - 1 ? 0 : b_ptr[sglid]) : BLOCK_READ_B(b_ptr, 0);
#endif
            b_ptr += input1_offset;
#elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            if (b_raw_global_id > N - 1) {
                b_tile[b_load_id] = 0;
            } else {
                uint b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + k * TILE_K), x);
                b_tile[b_load_id] = input1[b_idx];
            }
#endif // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
#else // IS_DYNAMIC
#if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
#if TILE_N_NOT_DIVISIBLE
            b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
#else // TILE_N_NOT_DIVISIBLE
            b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
#endif // TILE_N_NOT_DIVISIBLE
            b_ptr += input1_offset;
#elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            if (b_raw_global_id > N - 1) {
                b_tile[b_load_id] = 0;
            } else {
                uint b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + k * TILE_K), x);
                b_tile[b_load_id] = input1[b_idx];
            }
#endif // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
#endif // IS_DYNAMIC
        } // Loading B tile end
#if TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
        b_ptr = b_ptr + (input1_offset * sglid);
        b_tile = (N > b_raw_global_id) ? VLOAD(0, b_ptr) : 0;
        b_ptr = b_ptr + input1_offset1 - (input1_offset * sglid);
#endif // TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST

        // Loading A tile and tile C calculation
        unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
#if TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST
#if IS_DYNAMIC
#if HAS_DYNAMIC_K_PADDING || INPUT0_HAS_PADDING
            // In case of dynamic padding we can't guarantee memory access alignment for
            // block reads (4 bytes), so use scattered read
            uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (k * TILE_K + sglid));
            A_FLOATN a_read = input0[a_idx];
#else
            A_FLOATN a_read = TILE_K_NOT_DIVISIBLE ? a_ptr[sglid] : BLOCK_READ_A(a_ptr, 0);
#endif
#else // IS_DYNAMIC
#if TILE_K_NOT_DIVISIBLE
            A_FLOATN a_read = a_ptr[sglid];
#else // TILE_K_NOT_DIVISIBLE
            A_FLOATN a_read = BLOCK_READ_A(a_ptr, 0);
#endif // TILE_K_NOT_DIVISIBLE
#endif // IS_DYNAMIC
            a_ptr += input0_offset;

            unroll_for (uint subtile_k_id = 0; subtile_k_id < TILE_K / SIMD_WIDTH; subtile_k_id++) {
                unroll_for (uint simd_local_id = 0; simd_local_id < SIMD_WIDTH; simd_local_id++) {
#if TILE_K > SIMD_WIDTH
                    c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read[subtile_k_id], simd_local_id)),
                                         b_tile[subtile_k_id * SIMD_WIDTH + simd_local_id], c_tile[dot_id]);
#else // TILE_K > SIMD_WIDTH
                    c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_local_id)), b_tile[simd_local_id], c_tile[dot_id]);
#endif // TILE_K > SIMD_WIDTH
                }
            }
#elif TRANSPOSE_INPUT0 == TRANSPOSE_OTHER // TRANSPOSE_INPUT0
            uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (k * TILE_K + sglid));
            a_tile[dot_id] = input0[a_idx];
#endif // TRANSPOSE_INPUT0
        } // Loading A tile and tile C calculation end

#if TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST
        a_ptr = a_ptr + input0_offset1 - (input0_offset * tile_m_iterations);
#else // TRANSPOSE_INPUT0
    #if TRANSPOSE_INPUT0 == TRANSPOSE_Y_LAST
        a_ptr = a_ptr + (input0_offset * sglid);
        a_tile = VLOAD(0, a_ptr);
        a_ptr = a_ptr + input0_offset1 - (input0_offset * sglid);
    #endif

        // Tile C calculation for TN, TT cases
        unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
            unroll_for (uint simd_local_id = 0; simd_local_id < SIMD_WIDTH; simd_local_id++) {
                c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_tile[dot_id], simd_local_id)), b_tile[simd_local_id], c_tile[dot_id]);
            }
        } // Tile C calculation for TN, TT cases end
#endif // !TRANSPOSE_INPUT0

    } // Full tile calculation end

#if IS_DYNAMIC
    if (TILE_K_NOT_DIVISIBLE) {
        // Loading leftovers of the matrix B
        unroll_for (uint b_load_id = 0; b_load_id < TILE_K_LEFTOVER; b_load_id++) {
        #if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            #if HAS_DYNAMIC_N_PADDING || INPUT1_HAS_PADDING
                b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
            #else
                b_tile[b_load_id] = TILE_N_NOT_DIVISIBLE ? (b_raw_global_id > N - 1 ? 0 : b_ptr[sglid]) : BLOCK_READ_B(b_ptr, 0);
            #endif
            b_ptr += input1_offset;
        #elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == 0
            if (b_raw_global_id > N - 1) {
                b_tile[b_load_id] = 0;
            } else {
                uint b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + K_FULL_ITERATIONS * TILE_K), x);
                b_tile[b_load_id] = input1[b_idx];
            }
        #endif
        } // Loading leftovers of the matrix B end
        #if TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
            b_ptr = b_ptr + (input1_offset * sglid);
            b_tile = (N > b_raw_global_id) ? VLOAD(0, b_ptr) : 0;
        #endif // TRANSPOSE_INPUT1

        // Loading leftovers of the matrix A and tile C calculation
        unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
            uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (K_FULL_ITERATIONS * TILE_K + sglid));
            INPUT0_TYPE a_read = input0[a_idx];

            unroll_for (uint simd_id = 0; simd_id < TILE_K_LEFTOVER; simd_id++) {
                c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_id)), b_tile[simd_id], c_tile[dot_id]);
            }
        } // Loading leftovers of the matrix A and tile C calculation end
    }
#else // IS_DYNAMIC
#if TILE_K_NOT_DIVISIBLE
    // Loading leftovers of the matrix B
    unroll_for (uint b_load_id = 0; b_load_id < TILE_K_LEFTOVER; b_load_id++) {
    #if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
        #if TILE_N_NOT_DIVISIBLE
            b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
        #else // TILE_N_NOT_DIVISIBLE
            b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
        #endif // TILE_N_NOT_DIVISIBLE
        b_ptr += input1_offset;
    #elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == 0
        if (b_raw_global_id > N - 1) {
            b_tile[b_load_id] = 0;
        } else {
            uint b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + K_FULL_ITERATIONS * TILE_K), x);
            b_tile[b_load_id] = input1[b_idx];
        }
    #endif
    } // Loading leftovers of the matrix B end
    #if TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
        b_ptr = b_ptr + (input1_offset * sglid);
        b_tile = (N > b_raw_global_id) ? VLOAD(0, b_ptr) : 0;
    #endif // TRANSPOSE_INPUT1

    // Loading leftovers of the matrix A and tile C calculation
    unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
        uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (K_FULL_ITERATIONS * TILE_K + sglid));
        INPUT0_TYPE a_read = input0[a_idx];

        unroll_for (uint simd_id = 0; simd_id < TILE_K_LEFTOVER; simd_id++) {
            c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_id)), b_tile[simd_id], c_tile[dot_id]);
        }
    } // Loading leftovers of the matrix A and tile C calculation end
#endif // TILE_K_NOT_DIVISIBLE
#endif // IS_DYNAMIC

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
#if IS_DYNAMIC
    FUSED_OPS_PRELOAD_SCALAR;
#else // IS_DYNAMIC
#if TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
    FUSED_OPS_PRELOAD_SCALAR;
#else // TILE_N_NOT_DIVISIBLE
    FUSED_OPS_PRELOAD_VEC;
#endif // TILE_N_NOT_DIVISIBLE
#endif // IS_DYNAMIC
#endif // HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD

    // Writing result in the global memory
    unroll_for (uint write_id = 0; write_id < tile_m_iterations; write_id++) {
#if IS_DYNAMIC
        if (b_raw_global_id < N) {
#ifdef INPUT2_TYPE
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_ptr[sglid];
#else // INPUT2_TYPE
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_CALC_SCALAR;
#else // FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_SCALAR;
#endif // FUSED_OPS_CAN_USE_PRELOAD
            OUTPUT_TYPE res = FUSED_OPS_RESULT_SCALAR;
            *d_ptr = res;
#else // HAS_FUSED_OPS
            *d_ptr = dequantized;
#endif // HAS_FUSED_OPS
        }
#else // IS_DYNAMIC
#if TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
        if (b_raw_global_id < N) {
#ifdef INPUT2_TYPE
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_ptr[sglid];
#else // INPUT2_TYPE
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_CALC_SCALAR;
#else // FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_SCALAR;
#endif // FUSED_OPS_CAN_USE_PRELOAD
            OUTPUT_TYPE res = FUSED_OPS_RESULT_SCALAR;
            *d_ptr = res;
#else // HAS_FUSED_OPS
            *d_ptr = dequantized;
#endif // HAS_FUSED_OPS
        }

#else // TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1

#ifdef INPUT2_TYPE
        B_FLOATN c_val = BLOCK_READ_B(c_ptr, 0);
        ACCUMULATOR_TYPE_VEC dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_val;
#else // INPUT2_TYPE
        ACCUMULATOR_TYPE_VEC dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC_VEC;
#else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_VEC;
#endif // FUSED_OPS_CAN_USE_PRELOAD
        OUTPUT_TYPE_VEC res = FUSED_OPS_RESULT_VEC;
        BLOCK_WRITE_C(d_ptr, 0, res);
#else // HAS_FUSED_OPS
        BLOCK_WRITE_C(d_ptr, 0, dequantized);
#endif // HAS_FUSED_OPS

#endif // TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
#endif // IS_DYNAMIC
        d_ptr += batch_offset_output_diff;
#ifdef INPUT2_TYPE
        c_ptr += N;
#endif // INPUT2_TYPE
    } // Writing result in the global memory end
}

#undef BLOCK_SHUFFLE
#undef BLOCK_READ_A
#undef BLOCK_READ_B
#undef BLOCK_WRITE_C
#undef VLOAD
