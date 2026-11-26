// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/bf16_utils.cl"

// Feature-axis RMS on b_fs_yx_fsv16 / b_fs_zyx_fsv16 with F a multiple of 16. A work item column of FSPLIT sub-groups
// handles XBLK adjacent X positions: each sub-group block-reads whole fsv16 blocks (lane = feature within the block),
// the sum of squares is reduced across lanes and then across the FSPLIT sub-groups through SLM.

// gamma may have a different data type than the input
#define TO_GAMMA_ACC_TYPE(v) CAT(convert_, ACCUMULATOR_TYPE)(DECODE_INPUT1_COMPUTE_TYPE(v))

#if INPUT0_DIMS == 5
    #define FSV16_IN_IDX(fb)  INPUT0_GET_INDEX(b_idx, (fb) * SUB_GROUP_SIZE, z_idx, y_idx, x0)
    #define FSV16_OUT_IDX(fb) OUTPUT_GET_INDEX(b_idx, (fb) * SUB_GROUP_SIZE, z_idx, y_idx, x0)
#else
    #define FSV16_IN_IDX(fb)  INPUT0_GET_INDEX(b_idx, (fb) * SUB_GROUP_SIZE, y_idx, x0)
    #define FSV16_OUT_IDX(fb) OUTPUT_GET_INDEX(b_idx, (fb) * SUB_GROUP_SIZE, y_idx, x0)
#endif
#define FSV16_IN_VEC  MAKE_VECTOR_TYPE(INPUT0_TYPE, XBLK)
#define FSV16_OUT_VEC MAKE_VECTOR_TYPE(OUTPUT_TYPE, XBLK)
#define FSV16_READ_X(ptr, off)       BLOCK_READN(INPUT0_TYPE, XBLK, ptr, off)
#define FSV16_WRITE_X(ptr, off, val) BLOCK_WRITEN(OUTPUT_TYPE, XBLK, ptr, off, val)
#define fsv16_in_vec_u  CAT(rms_fsv16_in_vec_u_, KERNEL_ID)
#define fsv16_out_vec_u CAT(rms_fsv16_out_vec_u_, KERNEL_ID)
typedef union { FSV16_IN_VEC v; INPUT0_TYPE s[XBLK]; } fsv16_in_vec_u;
typedef union { FSV16_OUT_VEC v; OUTPUT_TYPE s[XBLK]; } fsv16_out_vec_u;

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(rms_gpu_fsv16)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
#if ELEMENTWISE_AFFINE
    const __global INPUT1_TYPE* gamma,
#endif
    __global OUTPUT_TYPE* output
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_DECLS
    #endif
)
{
    uint tile = (uint)get_global_id(0);
    const uint x0 = (tile % X_BLOCKS) * XBLK;
    tile /= X_BLOCKS;
    const uint y_idx = tile % INPUT0_SIZE_Y;
    tile /= INPUT0_SIZE_Y;
#if INPUT0_DIMS == 5
    const uint z_idx = tile % INPUT0_SIZE_Z;
    tile /= INPUT0_SIZE_Z;
#endif
    const uint b_idx = tile;

    const uint sg = (uint)get_local_id(1) / SUB_GROUP_SIZE;
    const uint lane = get_sub_group_local_id();

    float acc[XBLK];
    unroll_for (uint i = 0; i < XBLK; ++i) {
        acc[i] = 0.f;
    }
#if CACHE_INPUT
    fsv16_in_vec_u cache[BLOCKS_PER_SG];
#endif
    unroll_for (uint k = 0; k < BLOCKS_PER_SG; ++k) {
        const uint fb = sg + k * FSPLIT;
        if (fb < F_BLOCKS) {
            fsv16_in_vec_u v;
            v.v = FSV16_READ_X(input, FSV16_IN_IDX(fb));
#if CACHE_INPUT
            cache[k] = v;
#endif
            unroll_for (uint i = 0; i < XBLK; ++i) {
                const float t = DECODE_INPUT0_COMPUTE_TYPE(v.s[i]);
                acc[i] = fma(t, t, acc[i]);
            }
        }
    }
    unroll_for (uint i = 0; i < XBLK; ++i) {
        acc[i] = sub_group_reduce_add(acc[i]);
    }

#if FSPLIT > 1
    __local float partial[FSPLIT * XBLK];
    if (lane == 0) {
        unroll_for (uint i = 0; i < XBLK; ++i) {
            partial[sg * XBLK + i] = acc[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    unroll_for (uint i = 0; i < XBLK; ++i) {
        float s = 0.f;
        unroll_for (uint j = 0; j < FSPLIT; ++j) {
            s += partial[j * XBLK + i];
        }
        acc[i] = s;
    }
#endif

    float inv_rms[XBLK];
    unroll_for (uint i = 0; i < XBLK; ++i) {
        inv_rms[i] = rsqrt(acc[i] * INV_DATA_SIZE + EPSILON);
    }

    unroll_for (uint k = 0; k < BLOCKS_PER_SG; ++k) {
        const uint fb = sg + k * FSPLIT;
        if (fb < F_BLOCKS) {
            const uint f_base = fb * SUB_GROUP_SIZE;
#if CACHE_INPUT
            fsv16_in_vec_u v = cache[k];
#else
            fsv16_in_vec_u v;
            v.v = FSV16_READ_X(input, FSV16_IN_IDX(fb));
#endif
#if ELEMENTWISE_AFFINE
    #if RMS_GAMMA_IS_SCALAR
            const ACCUMULATOR_TYPE g = TO_GAMMA_ACC_TYPE(gamma[INPUT1_OFFSET]);
    #else
            const ACCUMULATOR_TYPE g = TO_GAMMA_ACC_TYPE(gamma[INPUT1_OFFSET + f_base + lane]);
    #endif
#else
            const ACCUMULATOR_TYPE g = 1.f;
#endif
            fsv16_out_vec_u o;
            unroll_for (uint xi = 0; xi < XBLK; ++xi) {
                OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(DECODE_INPUT0_COMPUTE_TYPE(v.s[xi]) * inv_rms[xi] * g);
#if HAS_FUSED_OPS
                FUSED_OPS;
                o.s[xi] = FUSED_OPS_RESULT;
#else
                o.s[xi] = normalized;
#endif
            }
            FSV16_WRITE_X(output, FSV16_OUT_IDX(fb), o.v);
        }
    }
}

#undef TO_GAMMA_ACC_TYPE
#undef FSV16_IN_IDX
#undef FSV16_OUT_IDX
#undef FSV16_IN_VEC
#undef FSV16_OUT_VEC
#undef FSV16_READ_X
#undef FSV16_WRITE_X
#undef fsv16_in_vec_u
#undef fsv16_out_vec_u
