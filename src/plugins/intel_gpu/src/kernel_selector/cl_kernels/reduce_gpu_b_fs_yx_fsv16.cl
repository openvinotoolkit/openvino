// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define SIMD 16
#define FSV 16

#if !defined REDUCE_BATCH
    #define REDUCE_BATCH 0
#endif
#if !defined REDUCE_FEATURE
    #define REDUCE_FEATURE 0
#endif
#if !defined REDUCE_Y
    #define REDUCE_Y 0
#endif
#if !defined REDUCE_X
    #define REDUCE_X 0
#endif

#define INPUT_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, READ_OFFSET)

#define ACCUMULATOR_VEC MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, READ_OFFSET)
#define TO_ACCUMULATOR_VEC CAT(convert_, ACCUMULATOR_VEC)
#define FINAL_ACCUMULATOR_VEC MAKE_VECTOR_TYPE(FINAL_ACCUMULATOR_TYPE, READ_OFFSET)

#define ACTIVATION_VEC MAKE_VECTOR_TYPE(ACTIVATION_TYPE, READ_OFFSET)
#define TO_ACTIVATION_VEC CAT(convert_, ACTIVATION_VEC)

#define OUTPUT_VEC MAKE_VECTOR_TYPE(OUTPUT_TYPE, READ_OFFSET)
#define TO_OUTPUT_VEC CAT(convert_, OUTPUT_VEC)

#define REDUCE_BFY_BY_FY_Y          REDUCE_BATCH && REDUCE_FEATURE && REDUCE_Y && !REDUCE_X || REDUCE_BATCH && REDUCE_Y && !REDUCE_FEATURE && !REDUCE_X || \
                                    REDUCE_FEATURE && REDUCE_Y && !REDUCE_BATCH && !REDUCE_X|| REDUCE_Y && !REDUCE_BATCH && !REDUCE_FEATURE && !REDUCE_X

#define REDUCE_F                    REDUCE_FEATURE && !REDUCE_BATCH && !REDUCE_Y && !REDUCE_X

#define NEED_SUB_GROUP_REDUCE       REDUCE_FEATURE

#if REDUCE_MAX_MODE
    #define INIT_VAL ACCUMULATOR_VAL_MIN
    #define INPUT_INIT_VAL INPUT0_VAL_MIN
#elif REDUCE_MIN_MODE
    #define INIT_VAL ACCUMULATOR_VAL_MAX
    #define INPUT_INIT_VAL INPUT0_VAL_MAX
#elif REDUCE_PROD_MODE || REDUCE_AND_MODE
    #define INIT_VAL ACCUMULATOR_VAL_ONE
    #define INPUT_INIT_VAL INPUT0_VAL_ONE
#else
    #define INIT_VAL ACCUMULATOR_VAL_ZERO
    #define INPUT_INIT_VAL INPUT0_VAL_ZERO
#endif

inline ACCUMULATOR_TYPE FUNC(apply_reduce)(ACCUMULATOR_TYPE acc, ACCUMULATOR_TYPE input) {
    #if REDUCE_SUM_MODE || REDUCE_MEAN_MODE || REDUCE_LOG_SUM_MODE
        acc += input;
    #elif REDUCE_MAX_MODE
        acc = ACCUMULATOR_MAX_FUNC(acc, input);
    #elif REDUCE_MIN_MODE
        acc = ACCUMULATOR_MIN_FUNC(acc, input);
    #elif REDUCE_PROD_MODE
        acc *= input;
    #elif REDUCE_AND_MODE
        acc = acc && input;
    #elif REDUCE_OR_MODE
        acc = acc || input;
    #elif REDUCE_SUM_SQUARE_MODE || REDUCE_L2_MODE
        acc += input * input;
    #elif REDUCE_L1_MODE
        #if !INPUT0_IS_FP
            acc += TO_ACCUMULATOR_TYPE(fabs(TO_FINAL_ACCUMULATOR_TYPE(input)));
        #else
            acc += fabs(input);
        #endif
    #elif REDUCE_LOG_SUM_EXP_MODE
        #if !INPUT0_IS_FP
            acc += TO_ACCUMULATOR_TYPE(exp(TO_FINAL_ACCUMULATOR_TYPE(input)));
        #else
            acc += exp(input);
        #endif
    #endif

    return acc;
}

inline ACCUMULATOR_TYPE FUNC(sub_group_reduce)(ACCUMULATOR_TYPE acc) {
    #if NEED_SUB_GROUP_REDUCE
        #if REDUCE_SUM_MODE || REDUCE_MEAN_MODE || REDUCE_LOG_SUM_MODE
            acc = sub_group_reduce_add(acc);
        #elif REDUCE_MAX_MODE
            acc = sub_group_reduce_max(acc);
        #elif REDUCE_MIN_MODE
            acc = sub_group_reduce_min(acc);
        #elif REDUCE_PROD_MODE
            ACCUMULATOR_TYPE next = ACCUMULATOR_VAL_ONE;
            acc *= _sub_group_shuffle_down(acc, next, 8);
            acc *= _sub_group_shuffle_down(acc, next, 4);
            acc *= _sub_group_shuffle_down(acc, next, 2);
            acc *= _sub_group_shuffle_down(acc, next, 1);
            acc  = _sub_group_shuffle(acc, 0);
        #elif REDUCE_AND_MODE
            acc = sub_group_all(acc);
        #elif REDUCE_OR_MODE
            acc = sub_group_any(acc);
        #elif REDUCE_SUM_SQUARE_MODE || REDUCE_L2_MODE
            acc = sub_group_reduce_add(acc);
        #elif REDUCE_L1_MODE
            acc = sub_group_reduce_add(acc);
        #elif REDUCE_LOG_SUM_EXP_MODE
            acc = sub_group_reduce_add(acc);
        #endif
    #endif

    return acc;
}

inline FINAL_ACCUMULATOR_TYPE FUNC(final_reduce)(FINAL_ACCUMULATOR_TYPE acc) {
    #if REDUCE_MEAN_MODE
        acc /= DIVIDER;
    #elif REDUCE_L2_MODE
        acc = sqrt(acc);
    #elif REDUCE_LOG_SUM_MODE || REDUCE_LOG_SUM_EXP_MODE
        acc = log(acc);
    #endif

    return acc;
}

inline uint FUNC(calc_linear_offset)(uint b, uint f, uint y, uint x) {
    uint index = b * COMMON_OUTPUT_SIZE_X * COMMON_OUTPUT_SIZE_Y * COMMON_OUTPUT_FEATURE_NUM +
                 f * COMMON_OUTPUT_SIZE_X * COMMON_OUTPUT_SIZE_Y +
                 y * COMMON_OUTPUT_SIZE_X +
                 x;

    return index;
}

REQD_SUB_GROUP_SIZE(SIMD)
KERNEL(reduce_fsv16)(
    const __global INPUT0_TYPE* data,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
)
{
#if IS_REDUCE_XY
    __local ACCUMULATOR_TYPE lg_storage[SIMD][BLOCK_Y_NUM];
    const uint lid0 = (uint)get_local_id(0);
    const uint lid1 = (uint)get_local_id(1);
    const uint x    = 0;
    const uint y    = 0;
#else
    const uint xy   = (uint)get_global_id(1) * READ_OFFSET;
    const uint x    = xy % ALIGN(COMMON_OUTPUT_SIZE_X, READ_OFFSET);
    const uint y    = xy / ALIGN(COMMON_OUTPUT_SIZE_X, READ_OFFSET);
#endif  // !IS_REDUCE_XY
    const uint bf   = (uint)get_global_id(2) * SIMD;
    const uint b    = bf / ALIGN(COMMON_OUTPUT_FEATURE_NUM, SIMD);
    const uint f    = bf % ALIGN(COMMON_OUTPUT_FEATURE_NUM, SIMD);

#if KEEP_DIMS
    const uint out_idx = OUTPUT_GET_INDEX(b, f, y, x);
#else
    #if REDUCE_BATCH && REDUCE_FEATURE && REDUCE_X                                      // BFX
        const uint out_idx = OUTPUT_GET_INDEX(y, x, b, f);
    #elif REDUCE_BATCH && REDUCE_FEATURE && REDUCE_Y                                    // BFY
        const uint out_idx = OUTPUT_GET_INDEX(x, b, f, y);
    #elif REDUCE_FEATURE && REDUCE_X                                                    // FX
        const uint out_idx = OUTPUT_GET_INDEX(b, y, f, x);
    #elif REDUCE_BATCH && REDUCE_X                                                      // BX
        const uint out_idx = OUTPUT_GET_INDEX(f + get_sub_group_local_id(), y, b, x);
    #elif REDUCE_BATCH && REDUCE_Y                                                      // BY
        const uint out_idx = OUTPUT_GET_INDEX(f + get_sub_group_local_id(), x, b, y);
    #elif REDUCE_FEATURE && REDUCE_Y                                                    // FY
        const uint out_idx = OUTPUT_GET_INDEX(b, x, f, y);
    #elif REDUCE_BATCH && REDUCE_FEATURE                                                // BF
        const uint out_idx = OUTPUT_GET_INDEX(y, x, b, f);
    #elif REDUCE_FEATURE                                                                // F
        const uint out_idx = OUTPUT_GET_INDEX(b + get_sub_group_local_id(), y, x, f);
    #elif REDUCE_BATCH                                                                  // B
        const uint out_idx = OUTPUT_GET_INDEX(f + get_sub_group_local_id(), y, x, b);
    #elif REDUCE_Y                                                                      // Y
        const uint out_idx = OUTPUT_GET_INDEX(b, f, x, y);
    #else
        const uint out_idx = OUTPUT_GET_INDEX(b, f, y, x);
    #endif
#endif

    const uint linear_idx = FUNC_CALL(calc_linear_offset)(b, f, y, x);
    if (linear_idx >= COMPUTATIONAL_OPERATIONS_NUMBER)
        return;

    const uint input_x_pitch = FSV;
    const uint input_y_pitch = input_x_pitch * (INPUT0_PAD_BEFORE_SIZE_X + INPUT0_SIZE_X + INPUT0_PAD_AFTER_SIZE_X);
    const uint input_fs_pitch = input_y_pitch * (INPUT0_PAD_BEFORE_SIZE_Y + INPUT0_SIZE_Y + INPUT0_PAD_AFTER_SIZE_Y);
    const uint input_batch_pitch = input_fs_pitch * ((INPUT0_PAD_BEFORE_FEATURE_NUM + INPUT0_FEATURE_NUM + INPUT0_PAD_AFTER_FEATURE_NUM + FSV - 1) / FSV);
    const uint padding_pitch = INPUT0_GET_INDEX(0, 0, 0, 0);

    const uint output_x_pitch = FSV;
    const uint output_y_pitch = FSV * (OUTPUT_PAD_BEFORE_SIZE_X + OUTPUT_SIZE_X + OUTPUT_PAD_AFTER_SIZE_X);

#if REDUCE_BATCH
    const uint batch_out = 0;
    const uint batch_max_val = INPUT0_BATCH_NUM;
#else
    const uint batch_out = BATCH_NUM_IDX_COMP(linear_idx);
    const uint batch_max_val = batch_out + 1;
#endif

#if REDUCE_FEATURE
    const uint feature_out = 0;
    const uint feature_max_val = INPUT0_FEATURE_NUM;
#else
    const uint feature_out = FEATURE_NUM_IDX_COMP(linear_idx);
    const uint feature_max_val = feature_out + 1;
#endif

#if REDUCE_Y
    #if IS_REDUCE_XY
    const uint y_out = (uint)get_local_id(1) * BLOCK_Y_SIZE;
    const uint y_max_val = min((uint)(y_out + BLOCK_Y_SIZE), (uint)INPUT0_SIZE_Y);
    #else
    const uint y_out = 0;
    const uint y_max_val = INPUT0_SIZE_Y;
    #endif
#else
    const uint y_out = SIZE_Y_IDX_COMP(linear_idx);
    const uint y_max_val = y_out + 1;
#endif

#if REDUCE_X
    const uint x_out = 0;
    const uint x_max_val = INPUT0_SIZE_X / READ_OFFSET;
    const uint x_leftover_start = x_max_val * READ_OFFSET;
    const uint x_leftover_end = INPUT0_SIZE_X;
#else
    const uint x_out = SIZE_X_IDX_COMP(linear_idx);
    const uint x_max_val = x_out + 1;
    const uint x_leftover_start = x_out;
    const uint x_leftover_end = x_max_val;
#endif

uint offset = batch_out * input_batch_pitch + ((feature_out + FSV - 1) / FSV) * input_fs_pitch + y_out * input_y_pitch + x_out * input_x_pitch + padding_pitch;

#if REDUCE_X
    ACCUMULATOR_TYPE acc = INIT_VAL;
    for (uint bi = batch_out; bi < batch_max_val; ++bi) {
        for (uint fi = feature_out; fi < feature_max_val; fi += FSV) {
            for (uint yi = y_out; yi < y_max_val; ++yi) {
                for (uint xi = x_out; xi < x_max_val; ++xi) {
                    INPUT_VEC input = (INPUT_VEC)(INPUT_INIT_VAL);
                    #if REDUCE_FEATURE && (INPUT0_FEATURE_NUM % FSV != 0) && !ZERO_INVARIANT_REDUCTION
                        if (fi + FSV <= INPUT0_FEATURE_NUM)
                            input = BLOCK_READ(data, offset);
                        else
                            if (fi + get_sub_group_local_id() < INPUT0_FEATURE_NUM)
                                for (int i = 0; i < READ_OFFSET; ++i)
                                    input[i] = data[offset + get_sub_group_local_id() + i * get_max_sub_group_size()];
                    #else
                        input = BLOCK_READ(data, offset);
                    #endif
                    unroll_for (int i = 0; i < READ_OFFSET; ++i)
                        acc = FUNC_CALL(apply_reduce)(acc, input[i]);
                    offset += input_x_pitch * READ_OFFSET;
                }
                #if INPUT0_SIZE_X % READ_OFFSET != 0
                    for (uint xi = x_leftover_start; xi < x_leftover_end; ++xi) {
                        INPUT0_TYPE leftovers = INIT_VAL;
                        #if REDUCE_FEATURE && (INPUT0_FEATURE_NUM % FSV != 0) && !ZERO_INVARIANT_REDUCTION
                            if (fi + FSV <= INPUT0_FEATURE_NUM)
                                leftovers = DT_INPUT_BLOCK_READ(data, offset);
                            else
                                if (fi + get_sub_group_local_id() < INPUT0_FEATURE_NUM)
                                    leftovers = data[offset + get_sub_group_local_id()];
                        #else
                            leftovers = DT_INPUT_BLOCK_READ(data, offset);
                        #endif
                        acc = FUNC_CALL(apply_reduce)(acc, leftovers);
                        offset += input_x_pitch;
                    }
                #endif
                offset += input_y_pitch - INPUT0_SIZE_X * input_x_pitch;
            }
            offset += input_fs_pitch - ((y_max_val - y_out) * input_y_pitch);
        }
        offset += input_batch_pitch - ((((feature_max_val - feature_out) + FSV - 1) / FSV) * input_fs_pitch);
    }

#if IS_REDUCE_XY
    lg_storage[lid0][lid1] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid1 != 0)
        return;

    #if REDUCE_SUM_SQUARE_MODE || REDUCE_L2_MODE || REDUCE_LOG_SUM_MODE || REDUCE_LOG_SUM_EXP_MODE
        acc = INIT_VAL;
        unroll_for (uint i = 0; i < BLOCK_Y_NUM; i++) {
            acc += lg_storage[lid0][i];
        }
    #else
        acc = lg_storage[lid0][0];
        unroll_for (uint i = 1; i < BLOCK_Y_NUM; i++) {
            acc = FUNC_CALL(apply_reduce)(acc, lg_storage[lid0][i]);
        }
    #endif
#endif

    FINAL_ACCUMULATOR_TYPE final_acc;
    acc = FUNC_CALL(sub_group_reduce)(acc);
    final_acc = FUNC_CALL(final_reduce)(TO_FINAL_ACCUMULATOR_TYPE(acc));
    OUTPUT_TYPE final_result;
    ACTIVATION_TYPE reduce_result = TO_ACTIVATION_TYPE(final_acc);
    #if HAS_FUSED_OPS
        FUSED_OPS_SCALAR;
        final_result = FUSED_OPS_RESULT_SCALAR;
    #else
        final_result = TO_OUTPUT_TYPE(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
    #endif

    #if (REDUCE_FEATURE && REDUCE_X || REDUCE_BATCH && REDUCE_X) && !KEEP_DIMS
        output[out_idx] = final_result;
    #elif REDUCE_BATCH && REDUCE_Y && REDUCE_X || REDUCE_BATCH && REDUCE_X || REDUCE_Y && REDUCE_X || REDUCE_X && !REDUCE_FEATURE
        DT_OUTPUT_BLOCK_WRITE(output + out_idx, 0, final_result);
    #else
        if (get_sub_group_local_id() == 0)
            output[out_idx] = final_result;
    #endif
#else  // !REDUCE_X
    ACCUMULATOR_VEC acc = (ACCUMULATOR_VEC)(INIT_VAL);
    for (uint bi = batch_out; bi < batch_max_val; ++bi) {
        for (uint fi = feature_out; fi < feature_max_val; fi += FSV) {

            for (uint yi = y_out; yi < y_max_val; ++yi) {
                for (uint xi = x_out; xi < x_max_val; ++xi) {
                    #if REDUCE_FEATURE && (INPUT0_FEATURE_NUM % FSV != 0) && !ZERO_INVARIANT_REDUCTION
                        INPUT_VEC input = (INPUT_VEC)(INPUT_INIT_VAL);
                        if (fi + FSV <= INPUT0_FEATURE_NUM)
                            input = BLOCK_READ(data, offset);
                        else
                            if (fi + get_sub_group_local_id() < INPUT0_FEATURE_NUM)
                                for (int i = 0; i < READ_OFFSET; ++i)
                                    input[i] = data[offset + get_sub_group_local_id() + i * get_max_sub_group_size()];
                    #else
                        INPUT_VEC input = BLOCK_READ(data, offset);
                    #endif

                    unroll_for (int i = 0; i < READ_OFFSET; ++i)
                        acc[i] = FUNC_CALL(apply_reduce)(acc[i], input[i]);
                    offset += input_x_pitch;
                }
                offset += input_y_pitch - (x_max_val - x_out) * input_x_pitch;
            }
            offset += input_fs_pitch - ((y_max_val - y_out) * input_y_pitch);
        }
        offset += input_batch_pitch - ((((feature_max_val - feature_out) + FSV - 1) / FSV) * input_fs_pitch);
    }

    FINAL_ACCUMULATOR_VEC final_acc;
    unroll_for (uint i = 0; i < READ_OFFSET; ++i) {
        acc[i] = FUNC_CALL(sub_group_reduce)(acc[i]);
        final_acc[i] = FUNC_CALL(final_reduce)(TO_FINAL_ACCUMULATOR_TYPE(acc[i]));
    }

    OUTPUT_VEC final_result;
    ACTIVATION_VEC reduce_result = TO_ACTIVATION_VEC(final_acc);

#if HAS_FUSED_OPS
    FUSED_OPS_VECTOR;
    final_result = (OUTPUT_VEC)(FUSED_OPS_RESULT_VECTOR);
#else
    final_result = TO_OUTPUT_VEC(ACTIVATION(reduce_result, ACTIVATION_PARAMS));
#endif

    unroll_for (uint i = 0; i < READ_OFFSET; ++i) {
        if(COMMON_OUTPUT_SIZE_X % READ_OFFSET == 0 || x + i < COMMON_OUTPUT_SIZE_X) {
            #if REDUCE_BATCH && REDUCE_FEATURE && REDUCE_Y && !REDUCE_X && !KEEP_DIMS
                output[out_idx + output_x_pitch * i] = final_result[i];
            #elif REDUCE_FEATURE && REDUCE_Y && !KEEP_DIMS
                if (get_sub_group_local_id() == 0)
                    output[out_idx + i] = final_result[i];
            #elif REDUCE_BATCH && REDUCE_Y && !KEEP_DIMS
                    output[out_idx + i] = final_result[i];
            #elif REDUCE_BATCH && REDUCE_Y && REDUCE_X && !KEEP_DIMS
                    output[out_idx + get_sub_group_local_id() + output_y_pitch * i] = final_result[i];
            #elif REDUCE_BFY_BY_FY_Y
                    output[out_idx + get_sub_group_local_id() + output_x_pitch * i] = final_result[i];
            #elif REDUCE_BATCH && REDUCE_FEATURE && !KEEP_DIMS
                if (get_sub_group_local_id() == 0)
                    output[out_idx + i] = final_result[i];
            #elif REDUCE_BATCH && !KEEP_DIMS
                    output[out_idx + output_y_pitch * i] = final_result[i];
            #elif REDUCE_BATCH && !REDUCE_FEATURE
                    DT_OUTPUT_BLOCK_WRITE(output + out_idx + output_x_pitch * i, 0, final_result[i]);
            #elif REDUCE_BATCH && REDUCE_FEATURE
                    if (get_sub_group_local_id() == 0)
                        output[out_idx + output_x_pitch * i] = final_result[i];
            #elif REDUCE_F && !KEEP_DIMS
                    if (get_sub_group_local_id() == 0)
                        output[out_idx + output_y_pitch * i] = final_result[i];
            #elif REDUCE_F
                    if (get_sub_group_local_id() == 0)
                        output[out_idx + output_x_pitch * i] = final_result[i];
            #endif
        }
    }
#endif  // !REDUCE_X
}

#undef SIMD
#undef FSV
#undef BLOCK_READ
#undef READ_OFFSET
#undef INPUT_VEC
#undef ACCUMULATOR_VEC
#undef TO_ACCUMULATOR_VEC
#undef FINAL_ACCUMULATOR_VEC
#undef ACTIVATION_VEC
#undef TO_ACTIVATION_VEC
#undef OUTPUT_VEC
#undef TO_OUTPUT_VEC
#undef REDUCE_BFY_BY_FY_Y
#undef REDUCE_F
#undef NEED_SUB_GROUP_REDUCE
#undef INIT_VAL
#undef INPUT_INIT_VAL
#undef REDUCE_BATCH
#undef REDUCE_FEATURE
#undef REDUCE_Y
#undef REDUCE_X
