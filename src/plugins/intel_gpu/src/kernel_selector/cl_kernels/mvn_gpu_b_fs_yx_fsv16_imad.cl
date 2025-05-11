// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/imad.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#include "mvn_gpu_b_fs_yx_fsv16_imad_accumulate.cl"
#include "mvn_gpu_b_fs_yx_fsv16_imad_reduce.cl"

// MVN - performs mean-variance normalization, that is normalizes the input data to have
//       0 mean and if NORMALIZE_VARIANCE is set to have variance 1.
//
// Below is a set of 5 kernels:
//   mvn_mean_1, mvn_mean_2, mvn_var_1, mvn_var_2, mvn_final
// that can perform mvn operation in two modes.
//
// Basic mode:
//   In this mode only mvn_final kernel is used. It performs required reductions for mean
//   and variance in this single kernel using single work-group for slice of data-sets
//   and reducing intermidiate values with local memory.
//   It does not require any additional jit constants.
//   lws:          LWS x 1 x 1
//   gws:          LWS x feature x batch
//
// Parallel mode:
//   In this mode all kernels are used to provide extra paralellism with global memory
//   and host side synchronization with evets/in-order queue.
//   To calculate mean:
//   mvn_mean_1 kernel should be first enqueued, provided extra global memory on second input
//     allowing to store intermidate results from all work-groups.
//     To activate this kernel MVN_KERNEL_MEAN_1 must be defined and evaluate to true/1.
//     lws:           LWS x 1 x 1
//     gws:           LWS * ITEM_GROUPS x feature x batch
//     This kernel will calculate partial results for each ITEM_GROUPS work-groups and store it into global memory.
//
//   mvn_mean_2 kernel must be next enqueued in order to further reduce previous results using single work-group.
//     This kernel expects on first input the result of mvn_mean_1 and on second input global memory of size
//     batch * align(feature, FSV) should be provided to store final mean values.
//     It needs to be ensured that mvn_mean_1 kernel has finished and stored its partial results into memory.
//     To activate this kernel MVN_KERNEL_MEAN_2 must be defined and evaluate to true/1.
//     lws:          LWS x 1 x 1
//     gws:          LWS x feature x batch
//
//  If required analogously the mvn_var_1 and mvn_var_2 kernels should be enqueud, additionally providing results from
//  mvn_mean_2 kernel.
//
//  Finally the mvn_final kernel should be enqueued with provided buffers with outputs from previous kernels
//  (mvn_mean_2, mvn_var_2). To enable parallel mode PRECALC_MEAN and optionally PRECALC_VARIANCE definitions should be
//  used. As at this stage there is no further need to synchronize and this kernel will perform simple normalization
//  given known mean and inverse of variance. Due to this this kernel can be enqueued with full paralellization, not
//  limiting it to single work-group.
//     lws:          SIMD x 1 x 1
//     gws:          (x * y) / SIMD * SIMD x feature x batch
//
// Required jit constants:
// SIMD         - Sub-group/simd size.
// LWS          - Local work-size along 0th dimension, must be multiple of SIMD.
// GWS          - Global work-size along 0th dimension.
//                In basic mode this must be equal to LWS.
//                In parallel mode this must be equal to LWS * ITEM_GROUPS, except in mvn_final kernel where it has no restrictions.
// ITEM_GROUPS  - Number of work-groups performing accumulation in parallel mode. Should be the same in both stages of parallel kernels.

#define FSV                   16
#define SG_NUM                (LWS / SIMD)

#define INPUT_TYPE2           MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
#define INPUT_TYPE4           MAKE_VECTOR_TYPE(INPUT0_TYPE, 4)
#define INPUT_TYPE8           MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
#define INPUT_PACKED_TYPE     MAKE_VECTOR_TYPE(INPUT0_TYPE, FSV)
#define OUTPUT_PACKED_TYPE    MAKE_VECTOR_TYPE(OUTPUT_TYPE, FSV)
#define MEAN_PACKED_TYPE      MAKE_VECTOR_TYPE(MEAN_TYPE, FSV)
#define INT_PACKED_TYPE       MAKE_VECTOR_TYPE(int, FSV)
#define ACC_PACKED_TYPE       MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, FSV)
#define ACT_PACKED_TYPE       MAKE_VECTOR_TYPE(ACTIVATION_TYPE, FSV)

#define TO_MEAN_PACKED_TYPE   CAT(convert_, MEAN_PACKED_TYPE)
#define TO_ACT_PACKED_TYPE    CAT(convert_, ACT_PACKED_TYPE)

#define ITEMS_NUM             (OUTPUT_SIZE_X * OUTPUT_SIZE_Y * OUTPUT_SIZE_Z)

// ================================================================================================
#if MVN_KERNEL_MEAN_1

DECLARE_PACKED_ACCUMULATE(accumulate_sum_input, ACCUMULATOR_TYPE, INPUT0_TYPE, FSV, INPUT_SLICE_PITCH, ITEMS_NUM, GWS, ACCUMULATE_SUM)

#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_sum_across_sg, ACCUMULATOR_TYPE, FSV, SG_NUM, REDUCE_NO_POST_OP)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_sum_inside_sg, ACCUMULATOR_TYPE, FSV, REDUCE_NO_POST_OP)
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(mvn_mean_1)(const __global INPUT0_TYPE* input,
                   __global ACCUMULATOR_TYPE* intermidiate_sum) {
    uint b = get_global_id(2);
    uint f = get_global_id(1) * FSV;
    uint flat_data_set_group = b * CEIL_DIV(OUTPUT_FEATURE_NUM, FSV) + get_global_id(1);

    uint items_group = get_group_id(0);
    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

#if INPUT0_DIMS == 5
    const uint data_sets_offset = INPUT0_GET_INDEX(b, f, 0, 0, 0);
#else  // INPUT0_DIMS == 4
    const uint data_sets_offset = INPUT0_GET_INDEX(b, f, 0, 0);
#endif

    ACC_PACKED_TYPE partial_sum = FUNC_CALL(accumulate_sum_input)(input, data_sets_offset, get_global_id(0));

#if SG_NUM != 1
    __local int slm_acc[(SG_NUM - 1) * FSV];
    int full_sum = FUNC_CALL(reduce_sum_across_sg)(partial_sum, slm_acc);
#else
    int full_sum = FUNC_CALL(reduce_sum_inside_sg)(partial_sum);
#endif

    if (sgid == 0 && (sglid < FSV || SIMD == FSV)) {
        intermidiate_sum[flat_data_set_group * ITEM_GROUPS * FSV + items_group * FSV + sglid] = full_sum;
    }
}
// ================================================================================================
#elif MVN_KERNEL_MEAN_2

DECLARE_PACKED_ACCUMULATE(accumulate_sum_input, ACCUMULATOR_TYPE, ACCUMULATOR_TYPE, FSV, FSV, ITEM_GROUPS, LWS, ACCUMULATE_SUM)

#define CALC_MEAN(sum) ((sum) / ITEMS_NUM)
#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_mean_across_sg, MEAN_TYPE, FSV, SG_NUM, CALC_MEAN)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_mean_inside_sg, MEAN_TYPE, FSV, CALC_MEAN)
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(mvn_mean_2)(const __global ACCUMULATOR_TYPE* intermidiate_sum,
                   __global MEAN_TYPE* intermidiate_mean) {
    uint b = get_global_id(2);
    uint f = get_global_id(1) * FSV;
    uint flat_data_set_group = b * CEIL_DIV(OUTPUT_FEATURE_NUM, FSV) + get_global_id(1);

    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint data_sets_offset = flat_data_set_group * ITEM_GROUPS * FSV;

    ACC_PACKED_TYPE complete_sum = FUNC_CALL(accumulate_sum_input)(intermidiate_sum, data_sets_offset, get_local_id(0));

#if SG_NUM != 1
    __local MEAN_TYPE slm_acc[(SG_NUM - 1) * FSV];
    MEAN_TYPE mean = FUNC_CALL(reduce_mean_across_sg)(TO_MEAN_PACKED_TYPE(complete_sum), slm_acc);
#else
    MEAN_TYPE mean = FUNC_CALL(reduce_mean_inside_sg)(TO_MEAN_PACKED_TYPE(complete_sum));
#endif

    if (sgid == 0 && (sglid < FSV || SIMD == FSV)) {
        intermidiate_mean[flat_data_set_group * FSV + sglid] = mean;
    }
}
// ================================================================================================
#elif MVN_KERNEL_VAR_1

#define EXTRA_ARGS_DECL_IMPL    , MEAN_TYPE mean
#define EXTRA_ARGS_IMPL         , mean
#define EXTRA_ARGS_DECL         EXTRA_ARGS_DECL_IMPL
#define EXTRA_ARGS              EXTRA_ARGS_IMPL
#define ACCUMULATE_SUM_SQ_DEV(curr, next, idx, mean)   ACCUMULATE_SUM_SQ(curr, TO_MEAN_TYPE(next) - _sub_group_shuffle(mean, idx), idx)
DECLARE_PACKED_ACCUMULATE_EARGS(accumulate_sum_sq_dev, MEAN_TYPE, INPUT0_TYPE, FSV, INPUT_SLICE_PITCH, ITEMS_NUM, GWS, ACCUMULATE_SUM_SQ_DEV, EXTRA_ARGS_DECL, EXTRA_ARGS)

#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_sum_across_sg, MEAN_TYPE, FSV, SG_NUM, REDUCE_NO_POST_OP)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_sum_inside_sg, MEAN_TYPE, FSV, REDUCE_NO_POST_OP)
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(mvn_var_1)(const __global INPUT0_TYPE* input,
                  const __global MEAN_TYPE* means,
                  __global MEAN_TYPE* intermidiate_sum) {
    uint b = get_global_id(2);
    uint f = get_global_id(1) * FSV;
    uint flat_data_set_group = b * CEIL_DIV(OUTPUT_FEATURE_NUM, FSV) + get_global_id(1);

    uint items_group = get_group_id(0);
    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

#if INPUT0_DIMS == 5
    const uint data_sets_offset = INPUT0_GET_INDEX(b, f, 0, 0, 0);
#else  // INPUT0_DIMS == 4
    const uint data_sets_offset = INPUT0_GET_INDEX(b, f, 0, 0);
#endif

    MEAN_TYPE mean = means[flat_data_set_group * FSV + sglid];
    MEAN_PACKED_TYPE partial_sum = FUNC_CALL(accumulate_sum_sq_dev)(input, data_sets_offset, get_global_id(0), mean);

#if SG_NUM != 1
    __local MEAN_TYPE slm_acc[(SG_NUM - 1) * FSV];
    MEAN_TYPE full_sum = FUNC_CALL(reduce_sum_across_sg)(partial_sum, slm_acc);
#else
    MEAN_TYPE full_sum = FUNC_CALL(reduce_sum_inside_sg)(partial_sum);
#endif

    if (sgid == 0 && (sglid < FSV || SIMD == FSV)) {
        intermidiate_sum[flat_data_set_group * ITEM_GROUPS * FSV + items_group * FSV + sglid] = full_sum;
    }
}
// ================================================================================================
#elif MVN_KERNEL_VAR_2

DECLARE_PACKED_ACCUMULATE(accumulate_sum, MEAN_TYPE, MEAN_TYPE, FSV, FSV, ITEM_GROUPS, LWS, ACCUMULATE_SUM)
#if defined EPS_OUTSIDE_SQRT
    #define CALC_INVERSE_VARIANCE(sum_diff_sq)   native_powr(native_sqrt((sum_diff_sq) / ITEMS_NUM) + (MEAN_TYPE)EPSILON, (MEAN_TYPE)-1.f);
#elif defined EPS_INSIDE_SQRT
    #define CALC_INVERSE_VARIANCE(sum_diff_sq)   native_powr((sum_diff_sq) / ITEMS_NUM + (MEAN_TYPE)EPSILON, (MEAN_TYPE)-0.5f)
#endif
#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_var_across_sg, MEAN_TYPE, FSV, SG_NUM, CALC_INVERSE_VARIANCE)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_var_inside_sg, MEAN_TYPE, FSV, CALC_INVERSE_VARIANCE)
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(mvn_var_2)(const __global MEAN_TYPE* intermidiate_sum,
                   __global MEAN_TYPE* intermidiate_ivar) {
    uint b = get_global_id(2);
    uint f = get_global_id(1) * FSV;
    uint flat_data_set_group = b * CEIL_DIV(OUTPUT_FEATURE_NUM, FSV) + get_global_id(1);

    uint items_group = get_group_id(0);
    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();

    const uint data_sets_offset = flat_data_set_group * ITEM_GROUPS * FSV;

    MEAN_PACKED_TYPE complete_sum = FUNC_CALL(accumulate_sum)(intermidiate_sum, data_sets_offset, get_local_id(0));

#if SG_NUM != 1
    __local MEAN_TYPE slm_acc[(SG_NUM - 1) * FSV];
    MEAN_TYPE inv_variance = FUNC_CALL(reduce_var_across_sg)(complete_sum, slm_acc);
#else
    MEAN_TYPE inv_variance = FUNC_CALL(reduce_var_inside_sg)(complete_sum);
#endif

    if (sgid == 0 && (sglid < FSV || SIMD == FSV)) {
        intermidiate_ivar[flat_data_set_group * FSV + sglid] = inv_variance;
    }
}

// ================================================================================================
#elif MVN_KERNEL_MAIN_BSV32

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(mvn_final_bsv32)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
    , const __global MEAN_TYPE* means
#if PRECALC_VARIANCE
    , const __global MEAN_TYPE* variances
#endif
) {
    uint b = get_global_id(2);
    uint f = get_global_id(1) * FSV;
    uint flat_data_set_group = b * CEIL_DIV(OUTPUT_FEATURE_NUM, FSV) + get_global_id(1);

    MEAN_PACKED_TYPE mean_vals = ((const __global MEAN_PACKED_TYPE*)(means + (flat_data_set_group * FSV)))[0];

#if PRECALC_VARIANCE
    MEAN_PACKED_TYPE inv_variance = ((const __global MEAN_PACKED_TYPE*)(variances + (flat_data_set_group * FSV)))[0];
#else // !PRECALC_VARIANCE
    MEAN_PACKED_TYPE inv_variance = (MEAN_PACKED_TYPE)(MEAN_VAL_ONE);
#endif

    if (b >= OUTPUT_BATCH_NUM || f >= OUTPUT_FEATURE_NUM)
        return;

    const uint output_spatial = get_global_id(0);
    uint x = output_spatial % OUTPUT_SIZE_X;
    uint y = output_spatial / OUTPUT_SIZE_X;
    uint input_offset = INPUT0_GET_INDEX(b, f, y, x);
    uint output_offset = OUTPUT_GET_INDEX(b, f, y, x);

    INPUT_PACKED_TYPE in_pack = ((const __global INPUT_PACKED_TYPE*)(input + input_offset))[0];
    ACT_PACKED_TYPE normalized_vec = fma((TO_ACT_PACKED_TYPE(in_pack) - TO_ACT_PACKED_TYPE(mean_vals)),
                                            TO_ACT_PACKED_TYPE(inv_variance), (ACT_PACKED_TYPE)0);
    OUTPUT_PACKED_TYPE result_vec = OUTPUT_VAL_ZERO;

    unroll_for (uint fi = 0; fi < FSV; fi++) {
        ACTIVATION_TYPE normalized = normalized_vec[fi];
#   if HAS_FUSED_OPS
        FUSED_OPS;
        result_vec[fi] = FUSED_OPS_RESULT;
#   else
        result_vec[fi] = TO_OUTPUT_TYPE(normalized);
#   endif
    }

    vstore16(result_vec, 0, &output[output_offset]);
}

#elif MVN_KERNEL_MEAN_VAR_BSV32

// Mean:
DECLARE_PACKED_ACCUMULATE(accumulate_sum_input, ACCUMULATOR_TYPE, INPUT0_TYPE, FSV, INPUT_SLICE_PITCH, ITEMS_NUM, LWS, ACCUMULATE_SUM)

#define CALC_MEAN(sum) ((sum) / ITEMS_NUM)
#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_mean, MEAN_TYPE, FSV, SG_NUM, CALC_MEAN)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_mean, MEAN_TYPE, FSV, CALC_MEAN)
#endif

// Variance:
#define EXTRA_ARGS_DECL_IMPL    , MEAN_TYPE mean
#define EXTRA_ARGS_IMPL         , mean
#define EXTRA_ARGS_DECL         EXTRA_ARGS_DECL_IMPL
#define EXTRA_ARGS              EXTRA_ARGS_IMPL
#define ACCUMULATE_SUM_SQ_DEV(curr, next, idx, mean)   ACCUMULATE_SUM_SQ(curr, TO_MEAN_TYPE(next) - _sub_group_shuffle(mean, idx), idx)
DECLARE_PACKED_ACCUMULATE_EARGS(accumulate_sum_sq_dev, MEAN_TYPE, INPUT0_TYPE, FSV, INPUT_SLICE_PITCH, ITEMS_NUM, LWS, ACCUMULATE_SUM_SQ_DEV, EXTRA_ARGS_DECL, EXTRA_ARGS)

#if defined EPS_OUTSIDE_SQRT
    #define CALC_INVERSE_VARIANCE(sum_diff_sq)   native_powr(native_sqrt((sum_diff_sq) / ITEMS_NUM) + (MEAN_TYPE)EPSILON, (MEAN_TYPE)-1.f);
#elif defined EPS_INSIDE_SQRT
    #define CALC_INVERSE_VARIANCE(sum_diff_sq)   native_powr((sum_diff_sq) / ITEMS_NUM + (MEAN_TYPE)EPSILON, (MEAN_TYPE)-0.5f)
#endif
#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_inverse_variance, MEAN_TYPE, FSV, SG_NUM, CALC_INVERSE_VARIANCE)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_inverse_variance, MEAN_TYPE, FSV, CALC_INVERSE_VARIANCE)
#endif

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(mvn_mean_var_bsv32)(
    const __global INPUT0_TYPE* input,
    __global MEAN_TYPE* means
#if NORMALIZE_VARIANCE
    , __global MEAN_TYPE* variances
#endif
) {
    uint b = get_global_id(2);
    uint f = get_global_id(1) * FSV;
    uint flat_data_set_group = b * CEIL_DIV(OUTPUT_FEATURE_NUM, FSV) + get_global_id(1);

    const uint sgid = get_sub_group_id();
    const uint sglid = get_sub_group_local_id();
    const uint data_sets_offset = INPUT0_GET_INDEX(b, f, 0, 0);

#if SG_NUM != 1
    __local MEAN_TYPE slm_acc[(SG_NUM - 1) * FSV];
#endif

    ACC_PACKED_TYPE partial_sum = FUNC_CALL(accumulate_sum_input)(input, data_sets_offset, get_local_id(0));
#if SG_NUM != 1
    MEAN_TYPE mean = FUNC_CALL(reduce_mean)(TO_MEAN_PACKED_TYPE(partial_sum), slm_acc);
#else
    MEAN_TYPE mean = FUNC_CALL(reduce_mean)(TO_MEAN_PACKED_TYPE(partial_sum));
#endif

#if NORMALIZE_VARIANCE
    MEAN_PACKED_TYPE partial_dev = FUNC_CALL(accumulate_sum_sq_dev)(input, data_sets_offset, get_local_id(0), mean);
    #if SG_NUM != 1
        MEAN_TYPE inv_variance = FUNC_CALL(reduce_inverse_variance)(partial_dev, slm_acc);
    #else
        MEAN_TYPE inv_variance = FUNC_CALL(reduce_inverse_variance)(partial_dev);
    #endif
#endif

    if (sgid == 0 && (sglid < FSV || SIMD == FSV)) {
        means[flat_data_set_group * FSV + sglid] = mean;
    #if NORMALIZE_VARIANCE
        variances[flat_data_set_group * FSV + sglid] = inv_variance;
    #endif
    }
}

// ================================================================================================
#else // MVN_KERNEL_MAIN

// Mean:
DECLARE_PACKED_ACCUMULATE(accumulate_sum_input, int, INPUT0_TYPE, FSV, INPUT_SLICE_PITCH, ITEMS_NUM, LWS, ACCUMULATE_SUM)

#define CALC_MEAN(sum) ((sum) / ITEMS_NUM)
#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_mean, MEAN_TYPE, FSV, SG_NUM, CALC_MEAN)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_mean, MEAN_TYPE, FSV, CALC_MEAN)
#endif

// Variance:
#define EXTRA_ARGS_DECL_IMPL    , MEAN_TYPE mean
#define EXTRA_ARGS_IMPL         , mean
#define EXTRA_ARGS_DECL         EXTRA_ARGS_DECL_IMPL
#define EXTRA_ARGS              EXTRA_ARGS_IMPL
#define ACCUMULATE_SUM_SQ_DEV(curr, next, idx, mean)   ACCUMULATE_SUM_SQ(curr, next - _sub_group_shuffle(mean, idx), idx)
DECLARE_PACKED_ACCUMULATE_EARGS(accumulate_sum_sq_dev, MEAN_TYPE, INPUT0_TYPE, FSV, INPUT_SLICE_PITCH, ITEMS_NUM, LWS, ACCUMULATE_SUM_SQ_DEV, EXTRA_ARGS_DECL, EXTRA_ARGS)

#if defined EPS_OUTSIDE_SQRT
    #define CALC_INVERSE_VARIANCE(sum_diff_sq)   native_powr(native_sqrt((sum_diff_sq) / ITEMS_NUM) + (MEAN_TYPE)EPSILON, (MEAN_TYPE)-1.f);
#elif defined EPS_INSIDE_SQRT
    #define CALC_INVERSE_VARIANCE(sum_diff_sq)   native_powr((sum_diff_sq) / ITEMS_NUM + (MEAN_TYPE)EPSILON, (MEAN_TYPE)-0.5f)
#endif
#if SG_NUM != 1
DECLARE_WG_PACKED_REDUCE_ADD(reduce_inverse_variance, MEAN_TYPE, FSV, SG_NUM, CALC_INVERSE_VARIANCE)
#else
DECLARE_SG_PACKED_REDUCE_ADD(reduce_inverse_variance, MEAN_TYPE, FSV, CALC_INVERSE_VARIANCE)
#endif

#define INPUT_PACKED_BLOCK_READ(ptr)   BLOCK_READN(INPUT0_TYPE, FSV, ptr, 0)

#define OUTPUT_PAD_IN_ITEMS (OUTPUT_PAD_BEFORE_SIZE_X != 0 || OUTPUT_PAD_AFTER_SIZE_X != 0 || OUTPUT_PAD_BEFORE_SIZE_Y != 0)

REQD_SUB_GROUP_SIZE(SIMD)
__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL(mvn_final)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
#if PRECALC_MEAN
    , const __global MEAN_TYPE* means
#endif
#if PRECALC_VARIANCE
    , const __global MEAN_TYPE* variances
#endif
) {
    uint b = get_global_id(2);
    uint f = get_global_id(1) * FSV;
    uint flat_data_set_group = b * CEIL_DIV(OUTPUT_FEATURE_NUM, FSV) + get_global_id(1);
#if GWS != LWS
    uint items_group = get_group_id(0);
#else
    uint items_group = 0;
#endif
    const uint sgid = get_sub_group_id() + items_group * SG_NUM;
    const uint sglid = get_sub_group_local_id();

#if INPUT0_DIMS == 5
    const uint data_sets_offset = INPUT0_GET_INDEX(b, f, 0, 0, 0);
#else  // INPUT0_DIMS == 4
    const uint data_sets_offset = INPUT0_GET_INDEX(b, f, 0, 0);
#endif
    uint input_offset;

#if (!PRECALC_MEAN || (NORMALIZE_VARIANCE && !PRECALC_VARIANCE)) && SG_NUM != 1
    __local MEAN_TYPE slm_acc[(SG_NUM - 1) * FSV];
#endif

#if PRECALC_MEAN
    MEAN_TYPE mean = means[flat_data_set_group * FSV + sglid];
#else
    INT_PACKED_TYPE partial_sum = FUNC_CALL(accumulate_sum_input)(input, data_sets_offset, get_local_id(0));
#   if SG_NUM != 1
    MEAN_TYPE mean = FUNC_CALL(reduce_mean)(TO_MEAN_PACKED_TYPE(partial_sum), slm_acc);
#   else
    MEAN_TYPE mean = FUNC_CALL(reduce_mean)(TO_MEAN_PACKED_TYPE(partial_sum));
#   endif
#endif

#if NORMALIZE_VARIANCE
#   if PRECALC_VARIANCE
    MEAN_TYPE inv_variance = variances[flat_data_set_group * FSV + sglid];
#   else
    MEAN_PACKED_TYPE partial_dev = FUNC_CALL(accumulate_sum_sq_dev)(input, data_sets_offset, get_local_id(0), mean);
#       if SG_NUM != 1
    MEAN_TYPE inv_variance = FUNC_CALL(reduce_inverse_variance)(partial_dev, slm_acc);
#       else
    MEAN_TYPE inv_variance = FUNC_CALL(reduce_inverse_variance)(partial_dev);
#       endif
#   endif
#else
    MEAN_TYPE inv_variance = 1;
#endif

#if OUTPUT_IS_FP
    input_offset = data_sets_offset + sgid * SIMD * FSV;
    uint output_spatial_base = sgid * SIMD;
#if OUTPUT_DIMS == 5
    uint output_offset = OUTPUT_GET_INDEX(b, f, 0, 0, 0) + sgid * SIMD * FSV;
#else  // OUTPUT_DIMS == 4
    uint output_offset = OUTPUT_GET_INDEX(b, f, 0, 0) + sgid * SIMD * FSV;
#endif
    // For fused ops to align with non-fp path
    const uint set_idx = sglid;

    for (uint spatial_idx = 0; spatial_idx < ITEMS_NUM / GWS; ++spatial_idx) {
        INPUT_PACKED_TYPE in_pack = INPUT_PACKED_BLOCK_READ(input + input_offset);

        unroll_for(uint si = 0; si < SIMD; ++si) {
            uint output_spatial = output_spatial_base + si;
            MEAN_TYPE normalized = (TO_MEAN_TYPE(in_pack[si]) - mean) * inv_variance;
            OUTPUT_TYPE result;
#           if HAS_FUSED_OPS
                FUSED_OPS;
                result = FUSED_OPS_RESULT;
#           else
                result = TO_OUTPUT_TYPE(normalized);
#           endif
#if !OUTPUT_PAD_IN_ITEMS
            DT_OUTPUT_BLOCK_WRITE(output, output_offset + si * SIMD, result);
#else
#   if OUTPUT_DIMS == 5
            uint z = output_spatial / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
            uint y = (output_spatial / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
            uint x = output_spatial % OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
#   else  // OUTPUT_DIMS == 4
            uint x = output_spatial % OUTPUT_SIZE_X;
            uint y = output_spatial / OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, y, x);
#   endif
            DT_OUTPUT_BLOCK_WRITE(output, output_offset, result);
#endif
        }
        input_offset += GWS * FSV;
        output_offset += GWS * FSV;
        output_spatial_base += GWS;
    }

    // [constexpr] Number of leftovers after full local work-group iterations.
    const uint lws_uniform_leftovers = ITEMS_NUM % GWS;
    // [constexpr] Number of sub-groups that can process leftovers loading SIMD items.
    const uint lws_uniform_leftovers_full_simds = lws_uniform_leftovers / SIMD;
    // [constexpr] Number of leftovers after full sub-group processing.
    const uint sg_uniform_leftovers = lws_uniform_leftovers % SIMD;

    if (lws_uniform_leftovers_full_simds > 0 && sgid < lws_uniform_leftovers_full_simds) {
        // Process leftovers that can use full sub-group.
        INPUT_PACKED_TYPE in_pack = INPUT_PACKED_BLOCK_READ(input + input_offset);

        unroll_for(uint si = 0; si < SIMD; ++si) {
            uint output_spatial = output_spatial_base + si;
            MEAN_TYPE normalized = (TO_MEAN_TYPE(in_pack[si]) - mean) * inv_variance;
            OUTPUT_TYPE result;
#           if HAS_FUSED_OPS
                FUSED_OPS;
                result = FUSED_OPS_RESULT;
#           else
                result = TO_OUTPUT_TYPE(normalized);
#           endif
#if !OUTPUT_PAD_IN_ITEMS
            DT_OUTPUT_BLOCK_WRITE(output, output_offset + si * SIMD, result);
#else
#   if OUTPUT_DIMS == 5
            uint z = output_spatial / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
            uint y = (output_spatial / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
            uint x = output_spatial % OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
#   else  // OUTPUT_DIMS == 4
            uint x = output_spatial % OUTPUT_SIZE_X;
            uint y = output_spatial / OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, y, x);
#   endif
            DT_OUTPUT_BLOCK_WRITE(output, output_offset, result);
#endif
        }
    } else if (lws_uniform_leftovers > 0 && sg_uniform_leftovers > 0 && sgid == lws_uniform_leftovers_full_simds) {
        // TODO: May be worth to consider the data here as across sub-group
        // Rest of leftovers, still use whole sub-group, but change addresses to not load extra data.
        INPUT_PACKED_TYPE in_pack;
        uint pack_idx = 0;
        if (sg_uniform_leftovers >= 8) {
            INPUT_TYPE8 tmp_in = DT_INPUT_BLOCK_READ8(input, input_offset + pack_idx * SIMD);
            in_pack[pack_idx + 0] = tmp_in[0];
            in_pack[pack_idx + 1] = tmp_in[1];
            in_pack[pack_idx + 2] = tmp_in[2];
            in_pack[pack_idx + 3] = tmp_in[3];
            in_pack[pack_idx + 4] = tmp_in[4];
            in_pack[pack_idx + 5] = tmp_in[5];
            in_pack[pack_idx + 6] = tmp_in[6];
            in_pack[pack_idx + 7] = tmp_in[7];
            pack_idx += 8;
        }
        if (sg_uniform_leftovers % 8 >= 4) {
            INPUT_TYPE4 tmp_in = DT_INPUT_BLOCK_READ4(input, input_offset + pack_idx * SIMD);
            in_pack[pack_idx + 0] = tmp_in[0];
            in_pack[pack_idx + 1] = tmp_in[1];
            in_pack[pack_idx + 2] = tmp_in[2];
            in_pack[pack_idx + 3] = tmp_in[3];
            pack_idx += 4;
        }
        if (sg_uniform_leftovers % 4 >= 2) {
            INPUT_TYPE2 tmp_in = DT_INPUT_BLOCK_READ2(input, input_offset + pack_idx * SIMD);
            in_pack[pack_idx + 0] = tmp_in[0];
            in_pack[pack_idx + 1] = tmp_in[1];
            pack_idx += 2;
        }
        if (sg_uniform_leftovers % 2 == 1) {
            in_pack[pack_idx] = DT_INPUT_BLOCK_READ(input, input_offset + pack_idx * SIMD);
        }

        OUTPUT_PACKED_TYPE result;
        unroll_for(uint si = 0; si < sg_uniform_leftovers; ++si) {
            uint output_spatial = output_spatial_base + si;
            MEAN_TYPE normalized = (TO_MEAN_TYPE(in_pack[si]) - mean) * inv_variance;
            OUTPUT_TYPE result;
#           if HAS_FUSED_OPS
                FUSED_OPS;
                result = FUSED_OPS_RESULT;
#           else
                result = TO_OUTPUT_TYPE(normalized);
#           endif
#if !OUTPUT_PAD_IN_ITEMS
            DT_OUTPUT_BLOCK_WRITE(output, output_offset + si * SIMD, result);
#else
#   if OUTPUT_DIMS == 5
            uint z = output_spatial / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
            uint y = (output_spatial / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
            uint x = output_spatial % OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
#   else  // OUTPUT_DIMS == 4
            uint x = output_spatial % OUTPUT_SIZE_X;
            uint y = output_spatial / OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, y, x);
#   endif
            DT_OUTPUT_BLOCK_WRITE(output, output_offset, result);
#endif
        }
    }
#else  // => !OUTPUT_IS_FP
    input_offset = data_sets_offset + sgid * SIMD * FSV;
#if OUTPUT_DIMS == 5
    uint output_offset = OUTPUT_GET_INDEX(b, f, 0, 0, 0) + sgid * SIMD * FSV;
#else  // OUTPUT_DIMS == 4
    uint output_offset = OUTPUT_GET_INDEX(b, f, 0, 0) + sgid * SIMD * FSV;
#endif
    uint output_spatial = sgid * SIMD + sglid;

    for (uint spatial_idx = 0; spatial_idx < ITEMS_NUM / GWS; ++spatial_idx) {
        INPUT_PACKED_TYPE in_pack = ((const __global INPUT_PACKED_TYPE*)(input + input_offset))[sglid];

        OUTPUT_PACKED_TYPE result;
        unroll_for(uint set_idx = 0; set_idx < FSV; ++set_idx) {
            MEAN_TYPE normalized = (TO_MEAN_TYPE(in_pack[set_idx]) - _sub_group_shuffle(mean, set_idx)) * _sub_group_shuffle(inv_variance, set_idx);
#           if HAS_FUSED_OPS
                FUSED_OPS;
                result[set_idx] = FUSED_OPS_RESULT;
#           else
                result[set_idx] = TO_OUTPUT_TYPE(normalized);
#           endif
        }
#if !OUTPUT_PAD_IN_ITEMS
        ((__global OUTPUT_PACKED_TYPE*)(output + output_offset))[sglid] = result;
#else
#   if OUTPUT_DIMS == 5
        uint z = output_spatial / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
        uint y = (output_spatial / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
        uint x = output_spatial % OUTPUT_SIZE_X;
        output_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
#   else  // OUTPUT_DIMS == 4
        uint x = output_spatial % OUTPUT_SIZE_X;
        uint y = output_spatial / OUTPUT_SIZE_X;
        output_offset = OUTPUT_GET_INDEX(b, f, y, x);
#   endif
        ((__global OUTPUT_PACKED_TYPE*)(output + output_offset))[0] = result;
#endif

        input_offset += GWS * FSV;
        output_offset += GWS * FSV;
        output_spatial += GWS;
    }

    // [constexpr] Number of leftovers after full local work-group iterations.
    const uint lws_uniform_leftovers = ITEMS_NUM % GWS;
    // [constexpr] Number of sub-groups that can process leftovers loading SIMD items.
    const uint lws_uniform_leftovers_full_simds = lws_uniform_leftovers / SIMD;
    // [constexpr] Number of leftovers after full sub-group processing.
    const uint sg_uniform_leftovers = lws_uniform_leftovers % SIMD;

    if (lws_uniform_leftovers_full_simds > 0 && sgid < lws_uniform_leftovers_full_simds) {
        // Process leftovers that can use full sub-group.
        INPUT_PACKED_TYPE in_pack = ((const __global INPUT_PACKED_TYPE*)(input + input_offset))[sglid];

        OUTPUT_PACKED_TYPE result;
        unroll_for(uint set_idx = 0; set_idx < FSV; ++set_idx) {
            MEAN_TYPE normalized = (TO_MEAN_TYPE(in_pack[set_idx]) - _sub_group_shuffle(mean, set_idx)) * _sub_group_shuffle(inv_variance, set_idx);
#           if HAS_FUSED_OPS
                FUSED_OPS;
                result[set_idx] = FUSED_OPS_RESULT;
#           else
                result[set_idx] = TO_OUTPUT_TYPE(normalized);
#           endif
        }
#if !OUTPUT_PAD_IN_ITEMS
        ((__global OUTPUT_PACKED_TYPE*)(output + output_offset))[sglid] = result;
#else
#   if OUTPUT_DIMS == 5
        uint z = output_spatial / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
        uint y = (output_spatial / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
        uint x = output_spatial % OUTPUT_SIZE_X;
        output_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
#   else  // OUTPUT_DIMS == 4
        uint x = output_spatial % OUTPUT_SIZE_X;
        uint y = output_spatial / OUTPUT_SIZE_X;
        output_offset = OUTPUT_GET_INDEX(b, f, y, x);
#   endif
        ((__global OUTPUT_PACKED_TYPE*)(output + output_offset))[0] = result;
#endif
    } else if (lws_uniform_leftovers > 0 && sg_uniform_leftovers > 0 && sgid == lws_uniform_leftovers_full_simds) {
        // TODO: May be worth to consider the data here as across sub-group
        // Rest of leftovers, still use whole sub-group, but change addresses to not load extra data.
        INPUT_PACKED_TYPE in_pack = ((const __global INPUT_PACKED_TYPE*)(input + input_offset))[sglid % sg_uniform_leftovers];

        OUTPUT_PACKED_TYPE result;
        unroll_for(uint set_idx = 0; set_idx < FSV; ++set_idx) {
            MEAN_TYPE normalized = (TO_MEAN_TYPE(in_pack[set_idx]) - _sub_group_shuffle(mean, set_idx)) * _sub_group_shuffle(inv_variance, set_idx);
#           if HAS_FUSED_OPS
                FUSED_OPS;
                result[set_idx] = FUSED_OPS_RESULT;
#           else
                result[set_idx] = TO_OUTPUT_TYPE(normalized);
#           endif
        }
        if (sglid < sg_uniform_leftovers) {
#if !OUTPUT_PAD_IN_ITEMS
            ((__global OUTPUT_PACKED_TYPE*)(output + output_offset))[sglid] = result;
#else
#   if OUTPUT_DIMS == 5
            uint z = output_spatial / (OUTPUT_SIZE_X * OUTPUT_SIZE_Y);
            uint y = (output_spatial / OUTPUT_SIZE_X) % OUTPUT_SIZE_Y;
            uint x = output_spatial % OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, z, y, x);
#   else  // OUTPUT_DIMS == 4
            uint x = output_spatial % OUTPUT_SIZE_X;
            uint y = output_spatial / OUTPUT_SIZE_X;
            output_offset = OUTPUT_GET_INDEX(b, f, y, x);
#   endif
            ((__global OUTPUT_PACKED_TYPE*)(output + output_offset))[0] = result;
#endif
        }
    }
#endif
}

#endif
// ================================================================================================

#undef FSV
#undef INPUT_SLICE_PITCH
#undef SG_NUM

#undef INPUT_TYPE2
#undef INPUT_TYPE4
#undef INPUT_TYPE8
#undef INPUT_PACKED_TYPE
#undef OUTPUT_PACKED_TYPE
#undef INT_PACKED_TYPE
#undef MEAN_PACKED_TYPE
#undef TO_MEAN_PACKED_TYPE

#undef INPUT_PACKED_BLOCK_READ
#undef OUTPUT_PAD_IN_ITEMS

#undef USE_IMAD
