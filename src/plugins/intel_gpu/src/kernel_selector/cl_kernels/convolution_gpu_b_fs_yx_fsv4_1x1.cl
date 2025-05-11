// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/fetch_weights.cl"
#include "include/batch_headers/imad.cl"

// ======================================================================================
// Host side jit-constants:
// ======================================================================================
// SIMD            { 16 } - number of work-items in sub-group/simd size;
//                          currently limited to only 16
// FEATURES_PER_WI { 16, 32 } - number of output features calculated by one
//                              work-item; must be multiple of SIMD
// LWG_DEPTH       { 1..16 } - number of sub-groups per work-group that will
//                             calculate the same output features, but accumulating
//                             different input features;
//                             helps in low EU utilization, but requires additional
//                             barrier and local memory reads/writes
// FORCE_PREFETCH { 0, 1 }   - flag to force the compiler to generate explicit
//                             data prefetching; requires additional global barrier
// ======================================================================================

#define FSV 4
#define WEIGHTS_OSV 16

#define DEQUANTIZED_TYPE float

#define INPUT_TYPE4       MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define FILTER_TYPE4      MAKE_VECTOR_TYPE(FILTER_TYPE, 4)
#define OUTPUT_TYPE4      MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4)
#define BIAS_TYPE4        MAKE_VECTOR_TYPE(BIAS_TYPE, 4)
#define DEQUANTIZED_TYPE4 MAKE_VECTOR_TYPE(DEQUANTIZED_TYPE, 4)

#define AS_INPUT_TYPE4(val)       CAT(as_, INPUT_TYPE4)(val)
#define AS_FILTER_TYPE4(val)      CAT(as_, FILTER_TYPE4)(val)
#define TO_DEQUANTIZED_TYPE(val)  CAT(convert_, DEQUANTIZED_TYPE)(val)
#define TO_DEQUANTIZED_TYPE4(val) CAT(convert_, DEQUANTIZED_TYPE4)(val)
#define TO_OUTPUT_TYPE4(val)      CAT(convert_, OUTPUT_TYPE4)(val)

#define GET_INPUT_INDEX(b, f, y, x)   GET_DATA_B_FS_YX_FSV4_INDEX(INPUT0, b, f, y, x)
#define GET_WEIGHTS_INDEX(b, f, y, x) GET_FILTER_OS_IS_YX_OSV16_ISV4_INDEX(FILTER, b, f, y, x)
#define GET_OUTPUT_INDEX(b, f, y, x)  GET_DATA_B_FS_YX_FSV4_INDEX(OUTPUT, b, f, y, x)

#define OUTPUT_FS_PITCH (OUTPUT_FEATURE_PITCH * FSV)
#define INPUT_FS_PITCH (INPUT0_FEATURE_PITCH * FSV)

#define WEIGHTS_IS_PITCH (WEIGHTS_OSV * FSV)
#define WEIGHTS_OS_PITCH ((FILTER_IFM_NUM + FSV - 1) / FSV * FSV * WEIGHTS_OSV)

#define MAX_SPATIAL_SIZE (OUTPUT_SIZE_X * OUTPUT_SIZE_Y)
#define SAFE_SPATIAL (MAX_SPATIAL_SIZE % SIMD == 0)
#define SAFE_FEATURES (OUTPUT_FEATURE_NUM % FEATURES_PER_WI == 0)

// Dispatch dimensions:
//     b x f               x spatial (y * x)
// WI: 1 x FEATURES_PER_WI x 1
// SG: 1 x FEATURES_PER_WI x SIMD

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(SIMD, 1, LWG_DEPTH)))
KERNEL(convolution)(
    const __global uint          *input,
    __global OUTPUT_TYPE4        *output,
    const __global int           *weights
#if BIAS_TERM
    , const __global BIAS_TYPE   *biases
#endif
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint f = (uint)get_global_id(1) * FEATURES_PER_WI;
#if LWG_DEPTH == 1
    const uint yx = (uint)get_global_id(0);
#else
    const uint yx = (uint)get_group_id(0) * SIMD + get_sub_group_local_id();
#endif
    const uint x = yx % OUTPUT_SIZE_X;
    const uint y = yx / OUTPUT_SIZE_X;
#if LWG_DEPTH == 1
    const uint b = (uint)get_global_id(2);
    const uint lwg_d = 0;
#else
    const uint b = get_group_id(2);
    const uint lwg_d = get_sub_group_id();
#endif

    int dotProd[FEATURES_PER_WI] = { 0 };

    int wei_sg[FEATURES_PER_WI / SIMD];
    int wei_sg_pre[FEATURES_PER_WI / SIMD];

    uint input_offset = GET_INPUT_INDEX(b, lwg_d * FSV, y * STRIDE_SIZE_Y, x * STRIDE_SIZE_X) / FSV;
    uint weights_offset = GET_WEIGHTS_INDEX(f + get_sub_group_local_id(), lwg_d * FSV, 0, 0) / FSV;

    // Prefetch input and weights
    uint in_u = input[input_offset];
    input_offset += INPUT_FS_PITCH / FSV * LWG_DEPTH;

    unroll_for (uint wi = 0; wi < (FEATURES_PER_WI / SIMD); ++wi) {
        const uint weights_os_offset = (wi * SIMD / WEIGHTS_OSV) * (WEIGHTS_OS_PITCH / FSV);
        const uint weights_osv_offset = (wi * SIMD % WEIGHTS_OSV) * (FSV / FSV);
        wei_sg_pre[wi] = weights[weights_offset + (weights_os_offset + weights_osv_offset)];
    }
    weights_offset += WEIGHTS_IS_PITCH / FSV * LWG_DEPTH;

#if FORCE_PREFETCH
    // Forces the compiler to emit prefetching send's before main loop.
    barrier(CLK_GLOBAL_MEM_FENCE);
#endif

    // Process four input features in one iteration - IMAD.
    for (uint k = 0; k < (FILTER_IFM_NUM + FSV - 1) / FSV / LWG_DEPTH; ++k) {
        INPUT_TYPE4 in_val = AS_INPUT_TYPE4(in_u);

        unroll_for (uint wi = 0; wi < (FEATURES_PER_WI / SIMD); ++wi) {
            wei_sg[wi] = wei_sg_pre[wi];
        }

        in_u = input[input_offset];
        input_offset += INPUT_FS_PITCH / FSV * LWG_DEPTH;

        unroll_for (uint wi = 0; wi < (FEATURES_PER_WI / SIMD); ++wi) {
            const uint weights_os_offset = (wi * SIMD / WEIGHTS_OSV) * (WEIGHTS_OS_PITCH / FSV);
            const uint weights_osv_offset = (wi * SIMD % WEIGHTS_OSV) * (FSV / FSV);
            wei_sg_pre[wi] = weights[weights_offset + (weights_os_offset + weights_osv_offset)];
        }
        weights_offset += WEIGHTS_IS_PITCH / FSV * LWG_DEPTH;

        unroll_for (uint out_fi = 0; out_fi < FEATURES_PER_WI; ++out_fi) {
            int wei_i = _sub_group_shuffle(wei_sg[out_fi / SIMD], out_fi % SIMD);
            FILTER_TYPE4 wei_val = AS_FILTER_TYPE4(wei_i);

            dotProd[out_fi] = IMAD(dotProd[out_fi], in_val, wei_val);
        }
    }

#if FORCE_PREFETCH
    // No thread should ever enter this, but forces the compiler to not split
    // weights/input prefetching instructions in main loop into conditional
    // block.
    if (f == OUTPUT_FEATURE_NUM) {
        dotProd[0] += 0 * in_u;
        unroll_for (uint out_fi = 0; out_fi < FEATURES_PER_WI; ++out_fi) {
            dotProd[out_fi] += 0 * wei_sg_pre[out_fi / SIMD];
        }
    }
#endif

#if LWG_DEPTH != 1
    // Accumulation is split across work-group.
    // Store partial results.
    local int lwg_acc[(LWG_DEPTH - 1) * SIMD * FEATURES_PER_WI];
    uint lwg_offset = (lwg_d - 1) * SIMD * FEATURES_PER_WI + get_sub_group_local_id();
    if (lwg_d != 0) {
        unroll_for (uint i = 0; i < FEATURES_PER_WI; ++i) {
            lwg_acc[lwg_offset] = dotProd[i];
            lwg_offset += SIMD;
        }
    }
    // Synchronize writes.
    barrier(CLK_LOCAL_MEM_FENCE);
    // Accumulate in first sub-group.
    if (lwg_d == 0) {
        lwg_offset = get_sub_group_local_id();
        unroll_for (uint j = 0; j < LWG_DEPTH - 1; ++j) {
            unroll_for (uint i = 0; i < FEATURES_PER_WI; ++i) {
                dotProd[i] += lwg_acc[lwg_offset];
                lwg_offset += SIMD;
            }
        }
    } else {
    // End other threads.
        return;
    }
#endif

    if (!SAFE_SPATIAL && yx >= MAX_SPATIAL_SIZE)
        return;

    uint output_offset_base = GET_OUTPUT_INDEX(b, f, y, x) / FSV;

    // TODO Handle output features % FSV != 0
    unroll_for (uint out_fi = 0; out_fi < FEATURES_PER_WI / FSV; ++out_fi) {
        if (!SAFE_FEATURES && f + out_fi * FSV >= OUTPUT_FEATURE_NUM)
            return;

        uint output_offset = output_offset_base + out_fi * (OUTPUT_FS_PITCH / FSV);
        DEQUANTIZED_TYPE4 dequantized;
        unroll_for (uint out_fii = 0; out_fii < FSV; ++out_fii) {
            dequantized[out_fii] = TO_DEQUANTIZED_TYPE(dotProd[out_fi * FSV + out_fii]);
        }

#if BIAS_TERM
        uint bias_offset = f + out_fi * FSV;
        BIAS_TYPE4 bias = ((const __global BIAS_TYPE4*)(biases + bias_offset))[0];
        dequantized += TO_DEQUANTIZED_TYPE4(bias);
#endif

    OUTPUT_TYPE4 out;

#if HAS_FUSED_OPS
        FUSED_OPS_PRELOAD;
        FUSED_OPS_CALC;
        out = TO_OUTPUT_TYPE4(FUSED_OPS_RESULT);
#else
        out = TO_OUTPUT_TYPE4(dequantized);
#endif
        if (OUTPUT_FEATURE_NUM % FSV != 0 && f + out_fi * FSV + FSV >= OUTPUT_FEATURE_NUM) {
            if (OUTPUT_FEATURE_NUM % FSV <= 1)
                out.s1 = (OUTPUT_TYPE)(0);
            if (OUTPUT_FEATURE_NUM % FSV <= 2)
                out.s2 = (OUTPUT_TYPE)(0);
            out.s3 = (OUTPUT_TYPE)(0);
        }
        output[output_offset] = out;
    }
}

#undef FSV
#undef WEIGHTS_OSV

#undef DEQUANTIZED_TYPE

#undef INPUT_TYPE4
#undef FILTER_TYPE4
#undef OUTPUT_TYPE4
#undef BIAS_TYPE4
#undef DEQUANTIZED_TYPE4

#undef AS_INPUT_TYPE4
#undef AS_FILTER_TYPE4
#undef TO_DEQUANTIZED_TYPE
#undef TO_DEQUANTIZED_TYPE4
#undef TO_OUTPUT_TYPE4

#undef GET_INPUT_INDEX
#undef GET_WEIGHTS_INDEX
#undef GET_OUTPUT_INDEX

#undef OUTPUT_FS_PITCH
#undef INPUT_FS_PITCH

#undef WEIGHTS_IS_PITCH
#undef WEIGHTS_OS_PITCH

#undef MAX_SPATIAL_SIZE
#undef SAFE_SPATIAL
#undef SAFE_FEATURES
