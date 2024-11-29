// Copyright (C) 2018-2024 Intel Corporation
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
#if BROADCAST_INPUT0
    DO_BROADCAST_INPUT0
#endif
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
#if BROADCAST_INPUT1
    DO_BROADCAST_INPUT1
#endif
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, y, x);
#else
#   error gemm_tiled_opt.cl : Unsupported input 1 format
#endif
}

inline uint FUNC(get_input1_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
    return FUNC_CALL(get_input1_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
}

#if BEAM_TABLE_TERM
inline uint FUNC(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if BEAM_TABLE_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(BEAM_TABLE, b, f, w, z, y, x);
#else
#   error gemm_tiled_ops.cl : Unsupported beam table format
#endif
}

inline uint FUNC(get_bt_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x) {
#if INDIRECT_INPUT0
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT0_DIMS_ORDER);
#else
    return FUNC_CALL(get_bt_index_nt)(OPTIONAL_SHAPE_INFO_TENSOR INPUT1_DIMS_ORDER);
#endif
}

#endif // BEAM_TABLE_TERM

#if INDIRECT_INPUT0
inline uint FUNC(get_input0_indirect_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x, __global BEAM_TABLE_TYPE* beam_table) {
#if INDIRECT_AXIS == 0
    int b_index = BEAM_TABLE_BATCH_NUM > 1 ? beam_table[FUNC_CALL(get_bt_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x)] : b;
#elif INDIRECT_AXIS == 1
    int b_index = BEAM_TABLE_FEATURE_NUM > 1 ? beam_table[FUNC_CALL(get_bt_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x)] : b;
#else
#   error gemm_tiled_opt.cl : Unsupported indirect axis for beam table
#endif
    return FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b_index, f, w, z, y, x);
}
#endif

#if INDIRECT_INPUT1
inline uint FUNC(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z, uint y, uint x, __global BEAM_TABLE_TYPE* beam_table) {
    int b_index = beam_table[FUNC_CALL(get_bt_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, x)];
    return FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b_index, f, w, z, y, x);
}
#endif

#ifdef BIAS_TERM
inline uint FUNC(get_input2_batch_offset)(OPTIONAL_SHAPE_INFO_ARG uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_tiled_opt.cl : Unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // BIAS_TERM

#define VLOAD CAT(vload, SIMD_WIDTH)

// Optimized gemm kernel for fp16/fp32 inputs
REQD_SUB_GROUP_SIZE(SIMD_WIDTH)
__attribute__((reqd_work_group_size(SIMD_WIDTH, 1, 1)))
KERNEL(gemm_tiled_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef BIAS_TERM
    const __global INPUT2_TYPE* input2,
#endif // BIAS_TERM
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
    #if B_VEC_SIZE == 1
    const uint x = (uint)get_global_id(0);
    #else
    const uint x = tile_n_num * SIMD_WIDTH * B_VEC_SIZE;
    #endif
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
#ifdef BIAS_TERM
    const uint batch_offset_input2 = FUNC_CALL(get_input2_batch_offset)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z);
#endif // BIAS_TERM
    uint y_write_id = 0;
    uint x_write_id = 0;
    const uint batch_offset_output = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR TR_B, TR_F, TR_W, TR_Z, TR_Y, TR_X);
    y_write_id = 1;
    x_write_id = 0;
    const uint output_y_pitch = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR TR_B, TR_F, TR_W, TR_Z, TR_Y, TR_X) - batch_offset_output;
    y_write_id = 0;
    x_write_id = 1;
    const uint output_x_pitch = FUNC_CALL(get_output_index)(OPTIONAL_SHAPE_INFO_TENSOR TR_B, TR_F, TR_W, TR_Z, TR_Y, TR_X) - batch_offset_output;

    // Start pointers offsets
#if TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST
    const __global INPUT0_TYPE* a_ptr = input0 + batch_offset_input0;
    #if INPUT0_HAS_DYNAMIC_PADDING || INPUT0_HAS_PADDING
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y+1), 0) - batch_offset_input0;
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, (TILE_K)) - batch_offset_input0;
    #else
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 1, 0);
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, (TILE_K));
    #endif
#elif TRANSPOSE_INPUT0 == TRANSPOSE_Y_LAST
    const __global INPUT0_TYPE* a_ptr = input0 + batch_offset_input0;
    #if INPUT0_HAS_DYNAMIC_PADDING || INPUT0_HAS_PADDING
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, 1) - batch_offset_input0;
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, (TILE_K)) - batch_offset_input0;
    #else
        const uint input0_offset = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, 1);
        const uint input0_offset1 = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, (TILE_K));
    #endif
#endif // TRANSPOSE_INPUT0
#if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
    const __global INPUT1_TYPE* b_ptr = input1 + batch_offset_input1;
    #if INPUT1_HAS_DYNAMIC_PADDING || INPUT1_HAS_PADDING
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, 1, tile_n_offset) - batch_offset_input1;
    #else
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 1, 0);
    #endif
#elif TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
    const __global INPUT1_TYPE* b_ptr = input1 + batch_offset_input1;
    #if INPUT1_HAS_DYNAMIC_PADDING || INPUT1_HAS_PADDING
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, 0, (tile_n_offset + 1)) - batch_offset_input1;
        const uint input1_offset1 = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (TILE_K), tile_n_offset) - batch_offset_input1;
    #else
        const uint input1_offset = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, 0, 1);
        const uint input1_offset1 = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR 0, 0, 0, 0, (TILE_K), 0);
    #endif
    const uint input1_fetch_size = ((N - tile_n_offset) < TILE_K) ? (N - tile_n_offset) : TILE_K;
#endif // TRANSPOSE_INPUT1
#ifdef BIAS_TERM
    const __global INPUT2_TYPE* c_ptr = input2 + batch_offset_input2 + tile_m_offset * N + tile_n_offset;
#endif // BIAS_TERM
    __global OUTPUT_TYPE* d_ptr = output + batch_offset_output;

    const uint b_raw_global_id = tile_n_offset + sglid;

#if INDIRECT_INPUT0 || INDIRECT_INPUT1
#if INDIRECT_AXIS == 0
    const char do_indirect_load = BEAM_TABLE_BATCH_NUM > 1;
#elif INDIRECT_AXIS == 1
    const char do_indirect_load = BEAM_TABLE_FEATURE_NUM > 1;
#else
#   error gemm_tiled_opt.cl : Unsupported indirect axis for beam table
#endif
#endif

#if TRANSPOSE_INPUT0 != TRANSPOSE_X_LAST
    MAKE_VECTOR_TYPE(INPUT0_TYPE, SIMD_WIDTH) a_tile;
#endif // TRANSPOSE_INPUT0 != TRANSPOSE_X_LAST
    B_FLOATN c_tile[TILE_M];

    unroll_for (uint i = 0; i < TILE_M; i++) {
        c_tile[i] = (B_FLOATN)(ACCUMULATOR_VAL_ZERO);
    }

    // Full tile calculation
    for (uint k = 0; k < K_FULL_ITERATIONS; k++) {
#if TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
        B_FLOATN b_tile[TILE_K];
#else // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
    #if B_VEC_SIZE == 1
        MAKE_VECTOR_TYPE(INPUT1_TYPE, SIMD_WIDTH) b_tile;
    #else
        MAKE_VECTOR_TYPE(INPUT1_TYPE, SIMD_WIDTH) b_tile[B_VEC_SIZE];
    #endif
#endif // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST

    // Loading B tile
#if (TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST)
        unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
    #if INDIRECT_INPUT1
            uint b_load_offset = (k * TILE_K) + b_load_id;
    #endif
    #if IS_DYNAMIC
        #if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            #if INDIRECT_INPUT1
            if (do_indirect_load)
            {
                #if B_VEC_SIZE == 1
                uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                b_tile[b_load_id] = b_raw_global_id >= N ? 0 : input1[b_idx];
                #else
                unroll_for(uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                    uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x + sglid + SIMD_WIDTH * b_elem, beam_table);
                    b_tile[b_load_id][b_elem] = b_raw_global_id + SIMD_WIDTH * b_elem >= N ? 0 : input1[b_idx];
                }
                #endif
            }
            else
            #endif // INDIRECT_INPUT1
            {
                #if B_VEC_SIZE == 1
                b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
                #else // B_VEC_SIZE == 1
                    if (TILE_N_NOT_DIVISIBLE == 0 || N_IS_ALIGNED_4BYTE)
                        b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
                    else {
                        unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                            b_tile[b_load_id][b_elem] = b_ptr[sglid + SIMD_WIDTH * b_elem];
                        }
                    }
                #endif // B_VEC_SIZE == 1
                b_ptr += input1_offset;
            }
        #elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            B_FLOATN b_tile[TILE_K];
            if (b_raw_global_id > N - 1) {
                b_tile[b_load_id] = 0;
            } else {
                uint b_idx = 0;
            #if INDIRECT_INPUT1
                if (do_indirect_load)
                {
                    b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                }
                else
            #endif // INDIRECT_INPUT1
                {
                    b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + k * TILE_K), x);
                }
                b_tile[b_load_id] = input1[b_idx];
            }
        #endif // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
    #else // IS_DYNAMIC
        #if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            #if INDIRECT_INPUT1
            if (do_indirect_load)
            {
                uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                b_tile[b_load_id] = b_raw_global_id >= N ? 0 : input1[b_idx];
            }
            else
            #endif // INDIRECT_INPUT1
            {
        #if N_IS_ALIGNED_4BYTE
                b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
        #else
                b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
        #endif
                b_ptr += input1_offset;
            }
        #elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            if (b_raw_global_id > N - 1) {
                b_tile[b_load_id] = 0;
            } else {
                uint b_idx = 0;
            #if INDIRECT_INPUT1
                if (do_indirect_load)
                {
                    b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                }
                else
            #endif // INDIRECT_INPUT1
                {
                    b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + k * TILE_K), x);
                }
                b_tile[b_load_id] = input1[b_idx];
            }
        #endif // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
    #endif // IS_DYNAMIC
    } // Loading B tile end
#elif TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
    #if INDIRECT_INPUT1
        if (do_indirect_load)
        {
        #if B_VEC_SIZE == 1
            unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
                uint b_load_offset = (k * TILE_K) + b_load_id;
                uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                b_tile[b_load_id] = b_raw_global_id >= N ? 0 : input1[b_idx];
            }
        #else
           unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
                    uint b_load_offset = k * TILE_K + b_load_id;
                    uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x + sglid + SIMD_WIDTH * b_elem, beam_table);
                    b_tile[b_elem][b_load_id] = b_raw_global_id + SIMD_WIDTH * b_elem >= N ? 0 : input1[b_idx];
                }
            }
        #endif
        }
        else
    #endif
        {
            b_ptr = b_ptr + (input1_offset * sglid);
        #if B_VEC_SIZE == 1
            b_tile = (N > b_raw_global_id) ? VLOAD(0, b_ptr) : 0;
        #else
            const __global INPUT1_TYPE* b_ptr_tmp = b_ptr;
            unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                b_tile[b_elem] = (N > b_raw_global_id + SIMD_WIDTH * b_elem) ? VLOAD(0, b_ptr_tmp) : 0;
                b_ptr_tmp += input1_offset * SIMD_WIDTH;
            }
        #endif // B_VEC_SIZE
            b_ptr = b_ptr + input1_offset1 - (input1_offset * sglid);
        }
#endif // TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST

        // Loading A tile and tile C calculation
#if IS_DYNAMIC && !INDIRECT_INPUT0 && !INPUT0_HAS_DYNAMIC_PADDING && !INPUT1_HAS_DYNAMIC_PADDING && TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST
        A_FLOATN a_read = (TILE_K_NOT_DIVISIBLE == 0 || K_IS_ALIGNED_4BYTE) ? BLOCK_READ_A(a_ptr, 0): a_ptr[sglid];
#endif
        unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
#if TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST
    #if IS_DYNAMIC
        #if INDIRECT_INPUT0
            uint a_idx = FUNC_CALL(get_input0_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (k * TILE_K + sglid), beam_table);
            A_FLOATN a_read = input0[a_idx];
        #elif INPUT0_HAS_DYNAMIC_PADDING || INPUT1_HAS_DYNAMIC_PADDING || INPUT0_HAS_PADDING
            // In case of dynamic padding we can't guarantee memory access alignment for
            // block reads (4 bytes), so use scattered read
            uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (k * TILE_K + sglid));
            A_FLOATN a_read = input0[a_idx];
        #endif
    #else // IS_DYNAMIC
        #if INDIRECT_INPUT0
            uint a_idx = FUNC_CALL(get_input0_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (k * TILE_K + sglid), beam_table);
            A_FLOATN a_read = input0[a_idx];
#elif K_IS_ALIGNED_4BYTE
            A_FLOATN a_read = BLOCK_READ_A(a_ptr, 0);
#else // K_IS_ALIGNED_4BYTE
            A_FLOATN a_read = a_ptr[sglid];
#endif // K_IS_ALIGNED_4BYTE
#endif // IS_DYNAMIC
            a_ptr += input0_offset;

            unroll_for (uint subtile_k_id = 0; subtile_k_id < TILE_K / SIMD_WIDTH; subtile_k_id++) {
                unroll_for (uint simd_local_id = 0; simd_local_id < SIMD_WIDTH; simd_local_id++) {
    #if TILE_K > SIMD_WIDTH
                    c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read[subtile_k_id], simd_local_id)),
                                         b_tile[subtile_k_id * SIMD_WIDTH + simd_local_id], c_tile[dot_id]);
    #else // TILE_K > SIMD_WIDTH
                #if B_VEC_SIZE > 1 && TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
                    MAKE_VECTOR_TYPE(INPUT1_TYPE, B_VEC_SIZE) b_tile_tmp;
                    unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                        b_tile_tmp[b_elem] = b_tile[b_elem][simd_local_id];
                    }
                    c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_local_id)), b_tile_tmp, c_tile[dot_id]);
                #else
                    c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_local_id)), b_tile[simd_local_id], c_tile[dot_id]);
                #endif
    #endif // TILE_K > SIMD_WIDTH
                }
            }
    #if IS_DYNAMIC && !INDIRECT_INPUT0 && !INPUT0_HAS_DYNAMIC_PADDING && !INPUT1_HAS_DYNAMIC_PADDING
        // Read A for next dot_id
        a_read = (dot_id + 1 < tile_m_iterations) ? (TILE_K_NOT_DIVISIBLE == 0 || K_IS_ALIGNED_4BYTE) ? BLOCK_READ_A(a_ptr, 0) : a_ptr[sglid] : 0;
    #endif
#elif TRANSPOSE_INPUT0 == TRANSPOSE_OTHER // TRANSPOSE_INPUT0
    #if INDIRECT_INPUT0
            uint a_idx = FUNC_CALL(get_input0_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (k * TILE_K + sglid), beam_table);
    #else
            uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (k * TILE_K + sglid));
    #endif
            a_tile[dot_id] = input0[a_idx];
#endif // TRANSPOSE_INPUT0
        } // Loading A tile and tile C calculation end

#if TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST
        a_ptr = a_ptr + input0_offset1 - (input0_offset * tile_m_iterations);
#else // TRANSPOSE_INPUT0
    #if TRANSPOSE_INPUT0 == TRANSPOSE_Y_LAST
        #if INDIRECT_INPUT0
            unroll_for (uint a_load_id = 0; a_load_id < SIMD_WIDTH; a_load_id++) {
                uint a_idx = FUNC_CALL(get_input0_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + a_load_id), (k * TILE_K + sglid), beam_table);
                a_tile[a_load_id] = input0[a_idx];
            }
        #else
            a_ptr = a_ptr + (input0_offset * sglid);
            a_tile = VLOAD(0, a_ptr);
            a_ptr = a_ptr + input0_offset1 - (input0_offset * sglid);
        #endif
    #endif
        // Tile C calculation for TN, TT cases
        unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
            unroll_for (uint simd_local_id = 0; simd_local_id < SIMD_WIDTH; simd_local_id++) {
            #if B_VEC_SIZE > 1 && TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
                MAKE_VECTOR_TYPE(INPUT1_TYPE, B_VEC_SIZE) b_tile_tmp;
                unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                    b_tile_tmp[b_elem] = b_tile[b_elem][simd_local_id];
                }
                c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_tile[dot_id], simd_local_id)), b_tile_tmp, c_tile[dot_id]);
            #else
                c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_tile[dot_id], simd_local_id)), b_tile[simd_local_id], c_tile[dot_id]);
            #endif
            }
        } // Tile C calculation for TN, TT cases end
#endif // !TRANSPOSE_INPUT0

    }
    // Full tile calculation end

    // Handle leftovers for K
#if IS_DYNAMIC
    if (TILE_K_NOT_DIVISIBLE) {
        // Loading leftovers of the matrix B
        #if TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
        B_FLOATN b_tile[TILE_K];
        #else // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
            #if B_VEC_SIZE == 1
        MAKE_VECTOR_TYPE(INPUT1_TYPE, SIMD_WIDTH) b_tile;
            #else
        MAKE_VECTOR_TYPE(INPUT1_TYPE, SIMD_WIDTH) b_tile[B_VEC_SIZE];
            #endif
        #endif // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
        unroll_for (uint b_load_id = 0; b_load_id < TILE_K_LEFTOVER; b_load_id++) {
        #if INDIRECT_INPUT1
            uint b_load_offset = (K_FULL_ITERATIONS * TILE_K) + b_load_id;
        #endif
        #if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            #if INDIRECT_INPUT1
            if (do_indirect_load)
                {
                #if B_VEC_SIZE == 1
                    uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                    b_tile[b_load_id] = b_raw_global_id >= N ? 0 : input1[b_idx];
                #else
                    unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                        uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x + sglid + SIMD_WIDTH * b_elem, beam_table);
                        b_tile[b_load_id][b_elem] = b_raw_global_id + SIMD_WIDTH * b_elem >= N ? 0 : input1[b_idx];
                    }
                #endif
                }
                else
            #endif // INDIRECT_INPUT1
                {
                #if B_VEC_SIZE == 1
                    b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
                #else // B_VEC_SIZE == 1
                    if (TILE_N_NOT_DIVISIBLE == 0 || N_IS_ALIGNED_4BYTE)
                        b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
                    else {
                        unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                            b_tile[b_load_id][b_elem] = b_ptr[sglid + SIMD_WIDTH * b_elem];
                        }
                    }
                #endif // B_VEC_SIZE == 1
                    b_ptr += input1_offset;
                }
        #elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == 0
        B_FLOATN b_tile[TILE_K];
        if (b_raw_global_id > N - 1) {
                b_tile[b_load_id] = 0;
            } else {
                uint b_idx = 0;
            #if INDIRECT_INPUT1
                if (do_indirect_load)
                {
                    b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                }
                else
            #endif
                {
                    b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + K_FULL_ITERATIONS * TILE_K), x);
                }
                b_tile[b_load_id] = input1[b_idx];
            }
        #endif // TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
        } // Loading leftovers of the matrix B end
        #if TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
            #if INDIRECT_INPUT1
                if (do_indirect_load)
                {
                #if B_VEC_SIZE == 1
                    unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
                        uint b_load_offset = (K_FULL_ITERATIONS * TILE_K) + b_load_id;
                        uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                        b_tile[b_load_id] = b_raw_global_id >= N ? 0 : input1[b_idx];
                    }
                #else
                    unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                        unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
                            uint b_load_offset = (K_FULL_ITERATIONS * TILE_K) + b_load_id;
                            uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x + sglid + SIMD_WIDTH * b_elem, beam_table);
                            b_tile[b_elem][b_load_id] = b_raw_global_id + SIMD_WIDTH * b_elem >= N ? 0 : input1[b_idx];
                        }
                    }
                #endif
                }
                else
            #endif
                {
                b_ptr = b_ptr + (input1_offset * sglid);
                #if B_VEC_SIZE == 1
                b_tile = (N > b_raw_global_id) ? VLOAD(0, b_ptr) : 0;
                #else
                const __global INPUT1_TYPE* b_ptr_tmp = b_ptr;
                unroll_for (uint b_elem = 0; b_elem < B_VEC_SIZE; ++b_elem) {
                    b_tile[b_elem] = (N > b_raw_global_id + SIMD_WIDTH * b_elem) ? VLOAD(0, b_ptr_tmp + input1_offset * SIMD_WIDTH * b_elem) : 0;
                }
                #endif
                b_ptr = b_ptr + input1_offset1 - (input1_offset * sglid);
                }
        #endif // TRANSPOSE_INPUT1

        // Loading leftovers of the matrix A and tile C calculation
        unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
        #if INDIRECT_INPUT0
            uint a_idx = FUNC_CALL(get_input0_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (K_FULL_ITERATIONS * TILE_K + sglid), beam_table);
        #else
            uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (K_FULL_ITERATIONS * TILE_K + sglid));
        #endif
            INPUT0_TYPE a_read = input0[a_idx];

            unroll_for (uint simd_id = 0; simd_id < TILE_K_LEFTOVER; simd_id++) {
            #if B_VEC_SIZE > 1
                #if TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
                MAKE_VECTOR_TYPE(INPUT1_TYPE, B_VEC_SIZE) b_tile_tmp = {b_tile[0][simd_id], b_tile[1][simd_id]};
                c_tile[dot_id] = mad((INPUT0_TYPE)sub_group_broadcast(a_read, simd_id), b_tile_tmp, c_tile[dot_id]);
                #else
                c_tile[dot_id] = mad((INPUT0_TYPE)sub_group_broadcast(a_read, simd_id), b_tile[simd_id], c_tile[dot_id]);
                #endif
            #else
                c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_id)), b_tile[simd_id], c_tile[dot_id]);
            #endif
            }
        } // Loading leftovers of the matrix A and tile C calculation end
    }
    #else // IS_DYNAMIC
    // Loading leftovers of the matrix B
        #if TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
        B_FLOATN b_tile[TILE_K];
        #else // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST
        MAKE_VECTOR_TYPE(INPUT1_TYPE, SIMD_WIDTH) b_tile;
        #endif // TRANSPOSE_INPUT1 != TRANSPOSE_Y_LAST

    unroll_for (uint b_load_id = 0; b_load_id < TILE_K_LEFTOVER; b_load_id++) {
        #if INDIRECT_INPUT1
        uint b_load_offset = (K_FULL_ITERATIONS * TILE_K) + b_load_id;
        #endif
        #if TRANSPOSE_INPUT1 == TRANSPOSE_X_LAST
            #if INDIRECT_INPUT1
            if (do_indirect_load)
            {
                uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                b_tile[b_load_id] = b_raw_global_id >= N ? 0 : input1[b_idx];
            }
            else
            #endif
            {
        #if N_IS_ALIGNED_4BYTE
                b_tile[b_load_id] = BLOCK_READ_B(b_ptr, 0);
        #else // N_IS_ALIGNED_4BYTE
                b_tile[b_load_id] = b_raw_global_id > N - 1 ? 0 : b_ptr[sglid];
        #endif // N_IS_ALIGNED_4BYTE
                b_ptr += input1_offset;
            }
        #elif TRANSPOSE_INPUT1 == TRANSPOSE_OTHER // TRANSPOSE_INPUT1 == 0
        if (b_raw_global_id > N - 1) {
            b_tile[b_load_id] = 0;
        } else {
            uint b_idx = 0;
            #if INDIRECT_INPUT1
            if (do_indirect_load)
            {
                b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
            }
            else
            #endif
            {
                b_idx = FUNC_CALL(get_input1_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (b_load_id + K_FULL_ITERATIONS * TILE_K), x);
            }
            b_tile[b_load_id] = input1[b_idx];
        }
        #endif
    } // Loading leftovers of the matrix B end
        #if TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST
            #if INDIRECT_INPUT1
         if (do_indirect_load) {
             unroll_for (uint b_load_id = 0; b_load_id < TILE_K; b_load_id++) {
                 uint b_load_offset = (K_FULL_ITERATIONS * TILE_K) + b_load_id;
                 uint b_idx = FUNC_CALL(get_input1_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, b_load_offset, x, beam_table);
                 b_tile[b_load_id] = b_raw_global_id >= N ? 0 : input1[b_idx];
             }
         }
         else
            #endif
         {
             b_ptr = b_ptr + (input1_offset * sglid);
             b_tile = (N > b_raw_global_id) ? VLOAD(0, b_ptr) : 0;
         }
        #endif // TRANSPOSE_INPUT1 == TRANSPOSE_Y_LAST

#if !INDIRECT_INPUT0 && K_IS_ALIGNED_4BYTE && (TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST)
    a_ptr = input0 + FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, y, (K_FULL_ITERATIONS * TILE_K));
#endif
    // Loading leftovers of the matrix A and tile C calculation
    unroll_for (uint dot_id = 0; dot_id < tile_m_iterations; dot_id++) {
        #if INDIRECT_INPUT0
        uint a_idx = FUNC_CALL(get_input0_indirect_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (K_FULL_ITERATIONS * TILE_K + sglid), beam_table);
        INPUT0_TYPE a_read = input0[a_idx];
#else  // INDIRECT_INPUT0
#if K_IS_ALIGNED_4BYTE && (TRANSPOSE_INPUT0 == TRANSPOSE_X_LAST)
        INPUT0_TYPE a_read = BLOCK_READ_A(a_ptr, 0);
        a_ptr += input0_offset;
#else
        uint a_idx = FUNC_CALL(get_input0_index)(OPTIONAL_SHAPE_INFO_TENSOR b, f, w, z, (y + dot_id), (K_FULL_ITERATIONS * TILE_K + sglid));
        INPUT0_TYPE a_read = input0[a_idx];
#endif
#endif // INDIRECT_INPUT0
        unroll_for (uint simd_id = 0; simd_id < TILE_K_LEFTOVER; simd_id++) {
            c_tile[dot_id] = mad((INPUT0_TYPE)(sub_group_broadcast(a_read, simd_id)), b_tile[simd_id], c_tile[dot_id]);
        }
    } // Loading leftovers of the matrix A and tile C calculation end
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
    #if B_VEC_SIZE == 1
        if (b_raw_global_id < N) {
        #ifdef BIAS_TERM
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_ptr[sglid];
        #else // BIAS_TERM
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
        #endif // BIAS_TERM

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
    #else
        #if TRANSPOSE_OUTPUT == TRANSPOSE_X_LAST
        const uint x_pitch = 1;
        #else
        const uint x_pitch = output_x_pitch;
        #endif
        OUTPUT_TYPE* d_ptr_tmp = d_ptr + sglid * x_pitch;

        #ifdef BIAS_TERM
        ACCUMULATOR_TYPE_VEC dequantized = (ACCUMULATOR_TYPE_VEC)(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_ptr[sglid];
        #else // BIAS_TERM
        ACCUMULATOR_TYPE_VEC dequantized = (ACCUMULATOR_TYPE_VEC)(ALPHA) * c_tile[write_id];
        #endif // BIAS_TERM
        #if HAS_FUSED_OPS
        FUSED_OPS_VEC;
        OUTPUT_TYPE_VEC result = FUSED_OPS_RESULT_VEC;
        unroll_for (uint n_elem = 0; n_elem < B_VEC_SIZE; ++n_elem) {
            if (b_raw_global_id + SIMD_WIDTH * n_elem < N) {
                *(d_ptr_tmp + SIMD_WIDTH * n_elem * x_pitch) = result[n_elem];
            }
        }
        #else
        unroll_for (uint n_elem = 0; n_elem < B_VEC_SIZE; ++n_elem) {
            if (b_raw_global_id + SIMD_WIDTH * n_elem < N) {
                *(d_ptr_tmp + SIMD_WIDTH * n_elem * x_pitch) = dequantized[n_elem];
            }
        }
        #endif // HAS_FUSED_OPS
    #endif // B_VEC_SIZE == 1
#else // IS_DYNAMIC
    #if TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
        if (b_raw_global_id < N) {
        #ifdef BIAS_TERM
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_ptr[sglid];
        #else // BIAS_TERM
            ACCUMULATOR_TYPE dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
        #endif // BIAS_TERM

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
        #ifdef BIAS_TERM
        B_FLOATN c_val = BLOCK_READ_B(c_ptr, 0);
        ACCUMULATOR_TYPE_VEC dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id] + TO_ACCUMULATOR_TYPE(BETA) * c_val;
        #else // BIAS_TERM
        ACCUMULATOR_TYPE_VEC dequantized = TO_ACCUMULATOR_TYPE(ALPHA) * c_tile[write_id];
        #endif // BIAS_TERM

        #if TRANSPOSE_OUTPUT == TRANSPOSE_X_LAST
        const uint x_pitch = 1;
        #else
        const uint x_pitch = output_x_pitch;
        #endif

        #if HAS_FUSED_OPS
            #if FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_CALC_VEC;
            #else // FUSED_OPS_CAN_USE_PRELOAD
        FUSED_OPS_VEC;
            #endif // FUSED_OPS_CAN_USE_PRELOAD
        OUTPUT_TYPE_VEC res = FUSED_OPS_RESULT_VEC;
        unroll_for (uint n_elem = 0; n_elem < B_VEC_SIZE; ++n_elem) {
            BLOCK_WRITEN(OUTPUT_TYPE, 1, d_ptr, SIMD_WIDTH * n_elem * output_x_pitch, res[n_elem]);
        }
        #else // HAS_FUSED_OPS
        unroll_for (uint n_elem = 0; n_elem < B_VEC_SIZE; ++n_elem) {
            BLOCK_WRITEN(OUTPUT_TYPE, 1, d_ptr, SIMD_WIDTH * n_elem * output_x_pitch, dequantized[n_elem]);
        }
        #endif // HAS_FUSED_OPS
    #endif // TILE_N_NOT_DIVISIBLE || B_VEC_SIZE == 1
#endif // IS_DYNAMIC
        d_ptr += output_y_pitch;
#ifdef BIAS_TERM
        c_ptr += N;
#endif // BIAS_TERM
    } // Writing result in the global memory end
}

#undef BLOCK_SHUFFLE
#undef BLOCK_READ_A
#undef BLOCK_READ_B
#undef BLOCK_WRITE_C
#undef VLOAD
