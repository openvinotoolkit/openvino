// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Multi-TILE_B dynamic kernel for fully_connected with INT4 compressed weights.
//
// Design:
//   - Dynamic shapes (shape-agnostic): batch unknown at compile time
//   - Precompiles 8 tile variants (FORCED_TILE_B=1..8)
//   - Runtime two-phase dispatch:
//     1) Main body: selects largest TILE_B in [8..4] that evenly divides batch.
//        If no exact divisor exists (primes, etc.), uses TILE_B=8.
//     2) Tail: remaining (batch % main_tile) elements dispatched with a smaller
//        tile variant. Zero-overhead when batch is exactly divisible.
//   - Each tile variant has NO batch-tail bounds checking in hot loops.
//   - INT4 compressed weights with per-group scale/zp decompression
//   - SIMD=16, supports os_is_yx_osv32_isv2, os_iyx_osv16, os_is_yx_osv64_isv2

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// =====================================================================
// Compile-time macros expected from C++ wrapper:
//   SIMD          - sub-group size (16)
//   TILE_OFM      - output feature tile {1, 2, 4}
//   TILE_IFM      - input feature tile {1, 2}
//   TILE_K        - weight elements per load {1, 2, 4}
//   TILE_K_OFM    - TILE_K * TILE_OFM
//   TILE_K_OFM_PACKED - TILE_K_OFM / 2 (for INT4)
//   IFM_SIZE, TILE_OUT_F_NUM, TILE_OUT_F_PITCH
//   TILE_IN_B_PITCH, TILE_OUT_B_PITCH
//   COMPRESSED_WEIGHTS_INT4, COMPRESSED_WEIGHTS
//   DECOMPRESSION_SCALE_TERM, DECOMPRESSION_ZP_TERM, etc.
//   W_IDX, TILE_OFM_PER_OSV_SIZE
//   DECOMPRESSION_SCALE_POST_OP (optional)
//   BATCH_SIZE - runtime value from shape_info
// =====================================================================

// Verify essential JIT parameters
#if SIMD != 16
#   error "fully_connected_gpu_bf_tiled_dyn_b.cl - SIMD must be 16"
#endif

#if TILE_OFM != 1 && TILE_OFM != 2 && TILE_OFM != 4
#   error "fully_connected_gpu_bf_tiled_dyn_b.cl - TILE_OFM must be one of {1, 2, 4}"
#endif

#if TILE_K != 1 && TILE_K != 2 && TILE_K != 4
#   error "fully_connected_gpu_bf_tiled_dyn_b.cl - TILE_K must be one of {1, 2, 4}"
#endif

#if !COMPRESSED_WEIGHTS_INT4
#   error "fully_connected_gpu_bf_tiled_dyn_b.cl - only INT4 compressed weights supported"
#endif

#if TILE_K_OFM != (TILE_K * TILE_OFM) || TILE_K_OFM > 8
#   error "fully_connected_gpu_bf_tiled_dyn_b.cl - TILE_K_OFM must be TILE_K * TILE_OFM and <= 8"
#endif

#if TILE_K_OFM != TILE_K_OFM_PACKED * 2
#   error "fully_connected_gpu_bf_tiled_dyn_b.cl - TILE_K_OFM must be divisible by 2 for INT4"
#endif

// INT4 unpack selection based on weight layout
#if TILE_K == 4 && FILTER_LAYOUT_OS_IS_YX_OSV32_ISV2
    #define UNPACK_INT4 UNPACK_INT4x2_OSV32_ISV2
#else
    #define UNPACK_INT4 UNPACK_INT4x2
#endif

// Vector type macros
#define INPUT_VEC_TYPE             MAKE_VECTOR_TYPE(INPUT0_TYPE, TILE_IFM)
#define ACCUMULATOR_VEC_TYPE       MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_OFM)
#define FILTER_VEC_TYPE            MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, TILE_K_OFM)
#define FILTER_PACKED_VEC_TYPE     MAKE_VECTOR_TYPE(FILTER_TYPE, TILE_K_OFM_PACKED)
#define BIAS_VEC_TYPE              MAKE_VECTOR_TYPE(BIAS_TYPE, TILE_OFM)
#define OUTPUT_VEC_TYPE            MAKE_VECTOR_TYPE(OUTPUT_TYPE, TILE_OFM)
#define ACTIVATION_VEC_TYPE        MAKE_VECTOR_TYPE(ACTIVATION_TYPE, TILE_OFM)
#define TO_OUTPUT_VEC_TYPE(x)      CAT(convert_, OUTPUT_VEC_TYPE)(x)
#define TO_ACTIVATION_VEC_TYPE(x)  CAT(convert_, ACTIVATION_VEC_TYPE)(x)
#define TO_FILTER_VEC_TYPE(x)      CAT(convert_, FILTER_VEC_TYPE)(x)
#define TO_ACCUMULATOR_VEC_TYPE(x) CAT(convert_, ACCUMULATOR_VEC_TYPE)(x)

// Block I/O macros
#define INPUT_BLOCK_READ(ptr, offset)        BLOCK_READN(INPUT0_TYPE, TILE_IFM, ptr, offset)
#define FILTER_BLOCK_READ(ptr, offset)       BLOCK_READN(FILTER_TYPE, TILE_K_OFM_PACKED, ptr, offset)
#define BIAS_BLOCK_READ(ptr, offset)         BLOCK_READN(BIAS_TYPE, TILE_OFM, ptr, offset)
#define OUTPUT_BLOCK_WRITE(ptr, offset, val) BLOCK_WRITEN(OUTPUT_TYPE, TILE_OFM, ptr, offset, val)

// Block write alignment check
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * TILE_OUT_B_PITCH) % 16 == 0 && (OUTPUT_TYPE_SIZE * OUTPUT_OFFSET) % 16 == 0)

#if !REALIGN_FP16_OFFSET
    #define MAIN_LOOP_ELEMENTS_COUNT  IFM_SIZE
#else
    #define MAIN_LOOP_ELEMENTS_COUNT  (IFM_SIZE - 1)
#endif

#define INPUT_ELEMENTS_COUNT IFM_SIZE

// =====================================================================
// Generate 8 tile variants (FORCED_TILE_B = 1..8)
// Each variant is a clean function with NO batch-tail bounds checking.
// =====================================================================
#pragma disable_includes_optimization

#define FORCED_TILE_B 1
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 2
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 3
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 4
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 5
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 6
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 7
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#define FORCED_TILE_B 8
#include "include/fully_connected_gpu_bf_tiled_dyn_b_core.cl"
#undef FORCED_TILE_B

#pragma enable_includes_optimization

// =====================================================================
// Argument forwarding macro for dispatching to tile variants
// =====================================================================
#define DYN_B_DISPATCH_ARGS \
    OPTIONAL_SHAPE_INFO_TENSOR \
    input, \
    DECOMPRESSION_SCALE_TERM_ARG \
    DECOMPRESSION_ZP_TERM_ARG \
    output, \
    weights \
    BIAS_TERM_ARG \
    FUSED_OPS_TERM_ARG

// Conditional argument macros
#if DECOMPRESSION_SCALE_TERM
    #define DECOMPRESSION_SCALE_TERM_ARG decompression_scale,
#else
    #define DECOMPRESSION_SCALE_TERM_ARG
#endif

#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    #define DECOMPRESSION_ZP_TERM_ARG decompression_zp,
#else
    #define DECOMPRESSION_ZP_TERM_ARG
#endif

#if BIAS_TERM
    #define BIAS_TERM_ARG , biases
#else
    #define BIAS_TERM_ARG
#endif

#if HAS_FUSED_OPS_DECLS
    #define FUSED_OPS_TERM_ARG , FUSED_OPS_ARGS
#else
    #define FUSED_OPS_TERM_ARG
#endif

// =====================================================================
// Main kernel entry: two-phase runtime TILE_B dispatch
//
// Phase 1 (main): Selects largest TILE_B in [8..4] that evenly divides
//   batch_size. If none exists (primes, etc.), uses TILE_B=8.
//   Processes floor(batch/main_tb) full tiles.
// Phase 2 (tail): Remaining batch%main_tb elements are dispatched to
//   a smaller tile variant. Only runs when batch is not exactly divisible.
//
// All branches are uniform (all work-items in a sub-group take the same
// path) so there is no SIMD divergence.
// =====================================================================
REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(fc_bf_tiled_dyn_b)(
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
) {
    const uint batch_size = BATCH_SIZE;
    const uint gid = (uint)get_group_id(0);

    // ---- Compute OFM tile from flat gid ----
    const uint ofm_tiles = CEIL_DIV(TILE_OUT_F_NUM, TILE_OFM * SIMD);
    const uint batch_tile = gid / ofm_tiles;
    const uint ofm_tile   = gid % ofm_tiles;
    const uint out_f = ofm_tile * (TILE_OFM * SIMD);

    // ---- Select main tile size (must match C++ select_tile_b logic) ----
    uint main_tb;
    if (batch_size <= 8) {
        main_tb = batch_size;
    } else {
        main_tb = 8;  // default for primes / no exact divisor in [4..8]
        if      (batch_size % 8 == 0) main_tb = 8;
        else if (batch_size % 7 == 0) main_tb = 7;
        else if (batch_size % 6 == 0) main_tb = 6;
        else if (batch_size % 5 == 0) main_tb = 5;
        else if (batch_size % 4 == 0) main_tb = 4;
    }

    const uint main_tiles = batch_size / main_tb;
    const uint tail = batch_size - main_tiles * main_tb;

    // ---- Phase 1: Main body (full tiles) ----
    if (batch_tile < main_tiles) {
        const uint out_b = batch_tile * main_tb;
        switch (main_tb) {
            case 1: FUNC_CALL(fc_dyn_b_tile1)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 2: FUNC_CALL(fc_dyn_b_tile2)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 3: FUNC_CALL(fc_dyn_b_tile3)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 4: FUNC_CALL(fc_dyn_b_tile4)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 5: FUNC_CALL(fc_dyn_b_tile5)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 6: FUNC_CALL(fc_dyn_b_tile6)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 7: FUNC_CALL(fc_dyn_b_tile7)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 8: FUNC_CALL(fc_dyn_b_tile8)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
        }
        return;
    }

    // ---- Phase 2: Tail (remaining elements, only when batch % main_tb != 0) ----
    if (batch_tile == main_tiles && tail > 0) {
        const uint out_b = main_tiles * main_tb;
        switch (tail) {
            case 1: FUNC_CALL(fc_dyn_b_tile1)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 2: FUNC_CALL(fc_dyn_b_tile2)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 3: FUNC_CALL(fc_dyn_b_tile3)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 4: FUNC_CALL(fc_dyn_b_tile4)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 5: FUNC_CALL(fc_dyn_b_tile5)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 6: FUNC_CALL(fc_dyn_b_tile6)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
            case 7: FUNC_CALL(fc_dyn_b_tile7)(out_b, out_f, DYN_B_DISPATCH_ARGS); break;
        }
    }
    // batch_tile > main_tiles + (tail>0): no work (excess work-items)
}

#undef INPUT_VEC_TYPE
#undef ACCUMULATOR_VEC_TYPE
#undef FILTER_VEC_TYPE
#undef FILTER_PACKED_VEC_TYPE
#undef BIAS_VEC_TYPE
#undef OUTPUT_VEC_TYPE
#undef ACTIVATION_VEC_TYPE
#undef TO_OUTPUT_VEC_TYPE
#undef TO_ACTIVATION_VEC_TYPE
#undef TO_FILTER_VEC_TYPE
#undef TO_ACCUMULATOR_VEC_TYPE

#undef INPUT_BLOCK_READ
#undef FILTER_BLOCK_READ
#undef BIAS_BLOCK_READ
#undef OUTPUT_BLOCK_WRITE

#undef USE_BLOCK_WRITE
#undef MAIN_LOOP_ELEMENTS_COUNT
#undef INPUT_ELEMENTS_COUNT
#undef UNPACK_INT4
