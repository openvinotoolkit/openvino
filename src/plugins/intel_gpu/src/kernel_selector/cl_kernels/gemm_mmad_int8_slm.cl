// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/mmad.cl"

#define ACCUMULATOR_TYPE_VEC    CAT(ACCUMULATOR_TYPE, SUB_GROUP_SIZE)
#define ACTIVATION_TYPE_VEC     CAT(ACTIVATION_TYPE, SUB_GROUP_SIZE)
#define PACKED_INPUT0_TYPE_VEC  CAT(PACKED_INPUT0_TYPE, SUB_GROUP_SIZE)
#define PACKED_INPUT1_TYPE_VEC  CAT(PACKED_INPUT1_TYPE, SUB_GROUP_SIZE)
#define BLOCK_READ(ptr)         _sub_group_block_read((const __global uint*)(ptr))

inline uint FUNC(get_input0_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT0_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT0, b, f, w, z, 0, 0);
#else // INPUT0_SIMPLE
#   error gemm_mmad_int8_slm.cl : Unsupported input 0 format
#endif // INPUT0_SIMPLE
}

inline uint FUNC(get_input1_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT1_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT1, b, f, w, z, 0, 0);
#else // INPUT1_SIMPLE
#   error gemm_mmad_int8_slm.cl : Unsupported input 1 format
#endif // INPUT1_SIMPLE
}

#ifdef INPUT2_TYPE
inline uint FUNC(get_input2_batch_offset)(uint b, uint f, uint w, uint z) {
#if INPUT2_SIMPLE
    return GET_DATA_INDEX_6D_SAFE(INPUT2, b, f, w, z, 0, 0);
#else // INPUT2_SIMPLE
#   error gemm_mmad_int8_slm.cl : Unsupported input 2 format
#endif // INPUT2_SIMPLE
}
#endif // INPUT2_TYPE

inline uint FUNC(get_output_batch_offset)(uint b, uint f, uint w, uint z) {
#if OUTPUT_SIMPLE
    return GET_DATA_INDEX_6D(OUTPUT, b, f, w, z, 0, 0);
#else // OUTPUT_SIMPLE
#   error gemm_mmad_int8_slm.cl : Unsupported output format
#endif // OUTPUT_SIMPLE
}

// GEMM int8 kernel using SLM and MMAD macro. Without transpositions of input matrices and without leftovers
__attribute__((reqd_work_group_size(SUB_GROUP_SIZE, PACK_SIZE, 1)))
REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(gemm_mmad_int8_slm)(
    const __global INPUT0_TYPE* input0,
    const __global INPUT1_TYPE* input1,
#ifdef INPUT2_TYPE
    const __global INPUT2_TYPE* input2,
#endif // INPUT2_TYPE
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif // HAS_FUSED_OPS_DECLS
    )
{
    // Indices
    const uint output_x = (uint)get_global_id(0);
    const uint output_x_tile = output_x * PACK_SIZE / SLM_TILE_SIZE;
    const uint output_y_tile = (uint)get_global_id(1);
#if HAS_FUSED_OPS
    uint output_y = output_y_tile * SUB_GROUP_SIZE;
#endif // HAS_FUSED_OPS

    uint batch = get_global_id(2);
    const uint lid0 = (uint)get_local_id(0);
    const uint lid1 = (uint)get_local_id(1);
    const uint gid0 = (uint)get_group_id(0);
    const uint gid1 = (uint)get_group_id(1);

    const uint z = batch % OUTPUT_SIZE_Z;
    batch /= OUTPUT_SIZE_Z;
    const uint w = batch % OUTPUT_SIZE_W;
    batch /= OUTPUT_SIZE_W;
    const uint f = batch % OUTPUT_FEATURE_NUM;
    batch /= OUTPUT_FEATURE_NUM;
    const uint b = batch % OUTPUT_BATCH_NUM;

    // Batch offsets
    const uint batch_offset_input0 = FUNC_CALL(get_input0_batch_offset)(b, f, w, z);
    const uint batch_offset_input1 = FUNC_CALL(get_input1_batch_offset)(b, f, w, z);
#ifdef INPUT2_TYPE
    const uint batch_offset_input2 = FUNC_CALL(get_input2_batch_offset)(b, f, w, z);
#endif // INPUT2_TYPE
    const uint batch_offset_output = FUNC_CALL(get_output_batch_offset)(b, f, w, z);

    // Pointer for loading the matrix B from the global memory
    __global PACKED_INPUT1_TYPE_VEC* input1_pnt = (__global PACKED_INPUT1_TYPE_VEC*)input1;

    // SLM tile of the matrix B
    __local PACKED_INPUT1_TYPE_VEC slm_tile_input1[SLM_TILE_SIZE * SLM_DECIMATION_FACTOR];

    // Pointer for loading the matrix B from SLM to registry chunks (GRF)
    __local INPUT1_TYPE* slm_tile_input1_pnt = (__local INPUT1_TYPE*)slm_tile_input1;

    // Registry chunks of input matrices (A, B) + input2 (optional)
    PACKED_INPUT0_TYPE_VEC reg_tile_input0;
    PACKED_INPUT1_TYPE_VEC reg_tile_input1;
#ifdef INPUT2_TYPE
    ACTIVATION_TYPE_VEC tile_input2;
#endif // INPUT2_TYPE

    // Registry chunks of the output matrix (C)
    ACCUMULATOR_TYPE_VEC reg_tile_output[4] = { (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO),
                                                (ACCUMULATOR_TYPE_VEC)(ACCUMULATOR_VAL_ZERO) };

    // Pointer to the result array (the matrix C)
    ACCUMULATOR_TYPE* reg_tile_output_pnt = (ACCUMULATOR_TYPE*)reg_tile_output;

    // Calculating positions for loading input matrices from the global memory
    const uint wg_offset_y = lid1 * SUB_GROUP_SIZE + lid0;
    const uint input1_index = (batch_offset_input1 + wg_offset_y * INPUT1_SIZE_X) / SLM_TILE_SIZE + gid0;
    const uint common_input0_offset = batch_offset_input0 + (gid1 * SLM_TILE_SIZE + lid1 * SUB_GROUP_SIZE) * INPUT0_SIZE_X;

#ifdef PRELOADING_SLM
    for (uint i = 0; i < SLM_DECIMATION_FACTOR; i++) {
        slm_tile_input1[i * SLM_TILE_SIZE + wg_offset_y] = input1_pnt[input1_index + i * INPUT1_SIZE_X];
    }

    // Synchronization; waiting until all work items will finish loading the matrix B from the global memory to SLM
    barrier(CLK_LOCAL_MEM_FENCE);
#endif // PRELOADING_SLM

    // Loop by "k" tiles
    for (uint k = 0; k < INPUT0_SIZE_X / SLM_TILE_SIZE; k++) {

        // Loading the matrix A from the global memory to GRF
        for (uint i = 0; i < SUB_GROUP_SIZE; i++) {
            reg_tile_input0[i] = AS_TYPE(PACKED_INPUT0_TYPE, BLOCK_READ(input0 + common_input0_offset + k * SLM_TILE_SIZE + i * INPUT0_SIZE_X));
        }

#ifndef PRELOADING_SLM
        // Loading the matrix B to SLM
        if (k % SLM_DECIMATION_FACTOR == 0) {

            // Synchronization
            barrier(CLK_LOCAL_MEM_FENCE);

            for (uint i = 0; i < SLM_DECIMATION_FACTOR; i++) {
                slm_tile_input1[i * SLM_TILE_SIZE + wg_offset_y] = input1_pnt[input1_index + (k + i) * INPUT1_SIZE_X];
            }

            // Synchronization; waiting until all work items will finish loading the matrix B from the global memory to SLM
            barrier(CLK_LOCAL_MEM_FENCE);
        }
#endif // PRELOADING_SLM

        // Loading the matrix B from SLM to GRF and calculating the matrix C
        MAKE_VECTOR_TYPE(INPUT1_TYPE, PACK_SIZE) temp_input1[SUB_GROUP_SIZE];

        // Here is 4 iterations in the extern loop because we should calculate 4 chunks of the matrix C
        for (uint i = 0; i < 4; i++) {
            const uint common_offset = (k % SLM_DECIMATION_FACTOR) * SLM_TILE_SIZE * SLM_TILE_SIZE + i * SUB_GROUP_SIZE + lid0;

            for (uint j = 0; j < SUB_GROUP_SIZE; j++) {
                temp_input1[j].s0 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE * (j * PACK_SIZE + 0)];
                temp_input1[j].s1 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE * (j * PACK_SIZE + 1)];
                temp_input1[j].s2 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE * (j * PACK_SIZE + 2)];
                temp_input1[j].s3 = slm_tile_input1_pnt[common_offset + SLM_TILE_SIZE * (j * PACK_SIZE + 3)];

                reg_tile_input1[j] = AS_TYPE(PACKED_INPUT1_TYPE, temp_input1[j]);
            }

        // Calculating one chunk of the matrix C
        reg_tile_output[i] = MMAD_8x8(reg_tile_input0, reg_tile_input1, reg_tile_output[i]);
        }
    } // End of the loop by "k"

#if HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD
    FUSED_OPS_PRELOAD;
#endif // HAS_FUSED_OPS && FUSED_OPS_CAN_USE_PRELOAD

    // Last calculations and writing result in the global memory
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < SUB_GROUP_SIZE; j++) {
            ACTIVATION_TYPE dequantized = TO_ACTIVATION_TYPE(reg_tile_output_pnt[i * SUB_GROUP_SIZE + j]);
            dequantized *= TO_ACTIVATION_TYPE(ALPHA);
#ifdef INPUT2_TYPE
            tile_input2[j] = TO_ACTIVATION_TYPE(input2[batch_offset_input2 + (output_y_tile * SUB_GROUP_SIZE + j) * OUTPUT_SIZE_X +
                                                       output_x_tile * SLM_TILE_SIZE + i * SUB_GROUP_SIZE + lid0]);
            dequantized += TO_ACTIVATION_TYPE(BETA) * tile_input2[j];
#endif // INPUT2_TYPE

#if HAS_FUSED_OPS
#if FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS_CALC;
#else // FUSED_OPS_CAN_USE_PRELOAD
            FUSED_OPS;
#endif // FUSED_OPS_CAN_USE_PRELOAD
            OUTPUT_TYPE res = FUSED_OPS_RESULT;
            output[batch_offset_output + (output_y_tile * SUB_GROUP_SIZE + j) * OUTPUT_SIZE_X +
                   output_x_tile * SLM_TILE_SIZE + i * SUB_GROUP_SIZE + lid0] = res;
            output_y++;
#else // HAS_FUSED_OPS
            output[batch_offset_output + (output_y_tile * SUB_GROUP_SIZE + j) * OUTPUT_SIZE_X +
                   output_x_tile * SLM_TILE_SIZE + i * SUB_GROUP_SIZE + lid0] = dequantized;
#endif // HAS_FUSED_OPS
        }
#if HAS_FUSED_OPS
        output_y -= SUB_GROUP_SIZE;
#endif // HAS_FUSED_OPS
    }
}

#undef ACCUMULATOR_TYPE_VEC
#undef ACTIVATION_TYPE_VEC
#undef PACKED_INPUT0_TYPE_VEC
#undef PACKED_INPUT1_TYPE_VEC
#undef BLOCK_READ
