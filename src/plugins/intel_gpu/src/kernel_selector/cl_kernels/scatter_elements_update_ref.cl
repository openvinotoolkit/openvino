// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_UPDATES_INDEX(idx_order) INPUT2_GET_INDEX(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#define GET_INPUT_INDEX(idx_order) INPUT0_GET_INDEX(idx_order)

#if AXIS_VALUE == 0
    #define SIZE INPUT0_BATCH_NUM
    #define ASSIGN_INDEX(index) b = index
#elif AXIS_VALUE == 1
    #define SIZE INPUT0_FEATURE_NUM
    #define ASSIGN_INDEX(index) f = index
#endif
#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
    #if AXIS_VALUE == 2
        #define SIZE INPUT0_SIZE_Y
        #define ASSIGN_INDEX(index) y = index
    #elif AXIS_VALUE == 3
        #define SIZE INPUT0_SIZE_X
        #define ASSIGN_INDEX(index) x = index
    #endif
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #if AXIS_VALUE == 2
        #define SIZE INPUT0_SIZE_Z
        #define ASSIGN_INDEX(index) z = index
    #elif AXIS_VALUE == 3
        #define SIZE INPUT0_SIZE_Y
        #define ASSIGN_INDEX(index) y = index
    #elif AXIS_VALUE == 4
        #define SIZE INPUT0_SIZE_X
        #define ASSIGN_INDEX(index) x = index
    #endif
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #if AXIS_VALUE == 2
        #define SIZE INPUT0_SIZE_W
        #define ASSIGN_INDEX(index) w = index
    #elif AXIS_VALUE == 3
        #define SIZE INPUT0_SIZE_Z
        #define ASSIGN_INDEX(index) z = index
    #elif AXIS_VALUE == 4
        #define SIZE INPUT0_SIZE_Y
        #define ASSIGN_INDEX(index) y = index
    #elif AXIS_VALUE == 5
        #define SIZE INPUT0_SIZE_X
        #define ASSIGN_INDEX(index) x = index
    #endif
#endif

#if OUTPUT_DIMS != INPUT2_DIMS
    #error "OUTPUT_DIMS is supposed to be same as INPUT2_DIMS"
#endif

#ifdef REDUCE_MODE
    #define SUM_MODE 1
    #define PROD_MODE 2
    #define MIN_MODE 3
    #define MAX_MODE 4
    #define MEAN_MODE 5
    #if INPUT2_IS_FP == 1 && INPUT2_TYPE_SIZE == 4   // f32
        #define FP_INT_MAX  MAXFLOAT
        #define FP_INT_MIN  (-MAXFLOAT)
    #elif INPUT2_IS_FP == 1                           // fp16
        #define FP_SCALE     65504.0f
        #define FP_SCALE_MAX 2147483648.0f
        #define FP_SCALE_MIN -FP_SCALE_MAX
        #define FP_INT_MAX  32767
        #define FP_INT_MIN  (-32767)
    #else                                             // int
        #define FP_INT_MAX  32767
        #define FP_INT_MIN  (-32767)
    #endif
    #define FP_INT_ZERO 0
    #define FP_INT_ONE  1

    #if REDUCE_MODE == SUM_MODE
        #define REDUCTION_NEUTRAL_VALUE FP_INT_ZERO
    #elif REDUCE_MODE == PROD_MODE
        #define REDUCTION_NEUTRAL_VALUE FP_INT_ONE
    #elif REDUCE_MODE == MIN_MODE
        #define REDUCTION_NEUTRAL_VALUE FP_INT_MAX
    #elif REDUCE_MODE == MAX_MODE
        #define REDUCTION_NEUTRAL_VALUE FP_INT_MIN
    #elif REDUCE_MODE == MEAN_MODE
        #define REDUCTION_NEUTRAL_VALUE FP_INT_ZERO
    #else
        #error "Invalid REDUCE_MODE value"
    #endif

    // Generic CAS loop. scope must match the storage qualifier of addr:
    //   memory_scope_work_group for __local,  memory_scope_device for __global.
    #define CAS_OP(addr, val, scope, expr) { \
        int expected_value; \
        int desired_value; \
        bool success; \
        do { \
            expected_value = atomic_load_explicit(addr, memory_order_acquire, scope); \
            desired_value  = (expr); \
            success = atomic_compare_exchange_weak_explicit(addr, &expected_value, desired_value, \
                          memory_order_acq_rel, memory_order_acquire, scope); \
        } while (!success); \
    }

    // to_int: encode INPUT2_TYPE value into int32 accumulator representation.
    // from_int: decode int32 accumulator back to float for output writeback.
    //
    // f32:  bit-reinterpret (as_int/as_float) — no scale, full f32 range.
    // fp16: fixed-point scale (FP_SCALE) — fp16 range fits within int32.
    // int:  identity — value is already an integer.
    inline int FUNC(to_int)(INPUT2_TYPE data_in)
    {
        #if INPUT2_IS_FP
            #if INPUT2_TYPE_SIZE == 4
                return as_int((float)data_in);
            #else
                float scaled = convert_float((half)data_in) * FP_SCALE;
                scaled = clamp(scaled, FP_SCALE_MIN, FP_SCALE_MAX);
                return convert_int_rte(scaled);
            #endif
        #else
            return data_in;
        #endif
    }

    inline float FUNC(from_int)(int acc)
    {
        #if INPUT2_IS_FP && INPUT2_TYPE_SIZE == 4
            return as_float(acc);
        #elif INPUT2_IS_FP
            return convert_float(acc) / FP_SCALE;
        #else
            return convert_float(acc);
        #endif
    }

    #if REDUCE_MODE < 1 || REDUCE_MODE > 5
        #error "Invalid REDUCE_MODE value"
    #endif

    // f32: native float atomics unavailable in OpenCL — use bit-reinterpret CAS for all modes.
    // fp16 / int: native atomic ops suffice for SUM/MIN/MAX/MEAN (fixed-point representation
    //             preserves ordering); only PROD requires a CAS loop.
    #if INPUT2_IS_FP == 1 && INPUT2_TYPE_SIZE == 4
        #if REDUCE_MODE == SUM_MODE || REDUCE_MODE == MEAN_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) \
                CAS_OP(addr, val, scope, as_int(as_float(expected_value) + as_float(val)))
        #elif REDUCE_MODE == MIN_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) \
                CAS_OP(addr, val, scope, as_int(fmin(as_float(expected_value), as_float(val))))
        #elif REDUCE_MODE == MAX_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) \
                CAS_OP(addr, val, scope, as_int(fmax(as_float(expected_value), as_float(val))))
        #elif REDUCE_MODE == PROD_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) \
                CAS_OP(addr, val, scope, as_int(as_float(expected_value) * as_float(val)))
        #endif
    #else
        #if REDUCE_MODE == SUM_MODE || REDUCE_MODE == MEAN_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) atomic_fetch_add(addr, val)
        #elif REDUCE_MODE == MIN_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) atomic_fetch_min(addr, val)
        #elif REDUCE_MODE == MAX_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) atomic_fetch_max(addr, val)
        #elif REDUCE_MODE == PROD_MODE && INPUT2_IS_FP == 1
            // fp16 PROD: decode fixed-point, multiply as float, re-encode.
            #define ATOMIC_REDUCE_OP(addr, val, scope) \
                CAS_OP(addr, val, scope, FUNC_CALL(to_int)( \
                    (convert_float(expected_value) / FP_SCALE) * (convert_float(val) / FP_SCALE)))
        #elif REDUCE_MODE == PROD_MODE
            #define ATOMIC_REDUCE_OP(addr, val, scope) \
                CAS_OP(addr, val, scope, expected_value * val)
        #endif
    #endif

    #define DEFINE_ATOMIC_REDUCE(STORAGE_QUALIFIER, SUFFIX, SCOPE) \
    inline void FUNC(atomic_reduce##SUFFIX)(volatile STORAGE_QUALIFIER int *ptr, int val) \
    { \
        atomic_int *atomic_addr = (atomic_int *)ptr; \
        ATOMIC_REDUCE_OP(atomic_addr, val, SCOPE); \
    }

    DEFINE_ATOMIC_REDUCE(__local,  _local,  memory_scope_work_group)
    DEFINE_ATOMIC_REDUCE(__global, _global, memory_scope_device)

    #define DEFINE_ATOMIC_COUNT(STORAGE_QUALIFIER, SUFFIX) \
    inline INPUT1_TYPE FUNC(count_add##SUFFIX)(volatile STORAGE_QUALIFIER int *ptr, int val) \
    { \
        atomic_int *atomic_addr = (atomic_int *)ptr; \
        return atomic_fetch_add(atomic_addr, val); \
    }

    DEFINE_ATOMIC_COUNT(__local, _local)
    DEFINE_ATOMIC_COUNT(__global, _global)
#endif

KERNEL(scatter_elements_update_ref)(OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
#if REDUCE_MODE != 0
                   , __global INPUT1_TYPE* output_fp
#if REDUCE_MODE == MEAN_MODE
                   , __global INPUT1_TYPE* count_k
#endif
#if ITER >= 1
#ifdef NO_LOCAL_MEMORY
                   , __global INPUT1_TYPE* reduction_v
#else
                   , __local INPUT1_TYPE* reduction_v
                   , __local INPUT1_TYPE* reduction_thread
#endif
#endif
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);
#if ITER == 0 // First kernel
    #if OUTPUT_DIMS == 4
        const uint x = dim0;
        const uint y = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint x = dim0 % OUTPUT_SIZE_X;
        const uint y = dim0 / OUTPUT_SIZE_X;
        const uint z = dim1 % OUTPUT_SIZE_Z;
        const uint w = dim1 / OUTPUT_SIZE_Z;
        const uint f = dim2 % OUTPUT_FEATURE_NUM;
        const uint b = dim2 / OUTPUT_FEATURE_NUM;
    #endif
    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[input_idx];

    #ifndef REDUCE_MODE
        #if HAS_FUSED_OPS
            FUSED_OPS_FIRST_KERNEL;
            output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
        #else
            output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
        #endif
    #else
        #if REDUCE_MODE == MEAN_MODE && USE_INIT_VAL == 0
            // For MEAN with USE_INIT_VAL=0: init to neutral so only scattered updates
            // contribute to the sum. Non-scattered positions restored in ITER=2.
            output_fp[output_idx] = FUNC_CALL(to_int)(REDUCTION_NEUTRAL_VALUE);
        #else
            INPUT1_TYPE val_fp = FUNC_CALL(to_int)(val);
            output_fp[output_idx] = val_fp;
        #endif
        #if REDUCE_MODE == MEAN_MODE
            count_k[output_idx] = INPUT1_VAL_ZERO;
        #endif
    #endif

#elif ITER == 1
    #ifdef REDUCE_MODE
        uint ORDER;
        #if OUTPUT_DIMS == 4
            x = dim0 % INPUT2_SIZE_X;
            y = dim0 / INPUT2_SIZE_X;
            f = dim1 % INPUT2_FEATURE_NUM;
            b = dim2 % INPUT2_BATCH_NUM;
        #elif OUTPUT_DIMS == 5
            x = dim0 % INPUT2_SIZE_X;
            y = dim0 / INPUT2_SIZE_X;
            z = dim1 % INPUT2_SIZE_Z;
            f = dim1 / INPUT2_SIZE_Z;
            b = dim2 % INPUT2_BATCH_NUM;
        #elif OUTPUT_DIMS == 6
            x = dim0 % INPUT2_SIZE_X;
            y = dim0 / INPUT2_SIZE_X;
            z = dim1 % INPUT2_SIZE_Z;
            w = dim1 / INPUT2_SIZE_Z;
            f = dim2 % INPUT2_FEATURE_NUM;
            b = dim2 / INPUT2_FEATURE_NUM;
        #endif
        const uint indices_idx = GET_INDICES_INDEX(ORDER);
        const uint updates_idx = GET_UPDATES_INDEX(ORDER);
        INPUT2_TYPE val = updates[(int)updates_idx];
        INPUT1_TYPE index = indices[(int)indices_idx];

        if (index < 0) { index += SIZE; }
        ASSIGN_INDEX(index);
        const uint output_idx = GET_OUTPUT_INDEX(ORDER);
        #ifndef NO_LOCAL_MEMORY
            reduction_thread[output_idx] = 0;
        #endif
        reduction_v[output_idx] = FUNC_CALL(to_int)(REDUCTION_NEUTRAL_VALUE);

        barrier(CLK_LOCAL_MEM_FENCE);

        int val_fixed = FUNC_CALL(to_int)(val);

        #ifndef NO_LOCAL_MEMORY
            INPUT1_TYPE write_thread = FUNC_CALL(count_add_local)(&reduction_thread[output_idx], 1);
            #if REDUCE_MODE != MEAN_MODE
                if (write_thread == 0) {
                    #if USE_INIT_VAL == 0
                        output_fp[output_idx] = FUNC_CALL(to_int)(REDUCTION_NEUTRAL_VALUE);
                    #endif
                }
            #endif
            if (reduction_thread[output_idx] > 1) {
                FUNC_CALL(atomic_reduce_local)(&reduction_v[output_idx], val_fixed);
            } else {
                reduction_v[output_idx] = val_fixed;
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            #if REDUCE_MODE == MEAN_MODE
                FUNC_CALL(count_add_global)(&count_k[output_idx], INPUT1_VAL_ONE);
            #endif

            if (write_thread == 0) {
                FUNC_CALL(atomic_reduce_global)(&output_fp[output_idx], reduction_v[output_idx]);
            }
        #else
            #if REDUCE_MODE == MEAN_MODE
                FUNC_CALL(count_add_global)(&count_k[output_idx], INPUT1_VAL_ONE);
            #endif

            #if REDUCE_MODE != MEAN_MODE && USE_INIT_VAL == 0
                output_fp[output_idx] = FUNC_CALL(to_int)(REDUCTION_NEUTRAL_VALUE);
            #endif

            FUNC_CALL(atomic_reduce_global)(&output_fp[output_idx], val_fixed);
        #endif
    #else // REDUCE_MODE==NONE.
        uint ORDER;
        #if OUTPUT_DIMS == 4
            x = dim0;
            y = dim1;
            f = dim2 % INPUT2_FEATURE_NUM;
            b = dim2 / INPUT2_FEATURE_NUM;
        #elif OUTPUT_DIMS == 5
            x = dim0 % INPUT2_SIZE_X;
            y = dim0 / INPUT2_SIZE_X;
            z = dim1;
            f = dim2 % INPUT2_FEATURE_NUM;
            b = dim2 / INPUT2_FEATURE_NUM;
        #elif OUTPUT_DIMS == 6
            x = dim0 % INPUT2_SIZE_X;
            y = dim0 / INPUT2_SIZE_X;
            z = dim1 % INPUT2_SIZE_Z;
            w = dim1 / INPUT2_SIZE_Z;
            f = dim2 % INPUT2_FEATURE_NUM;
            b = dim2 / INPUT2_FEATURE_NUM;
        #endif
        const uint indices_idx = GET_INDICES_INDEX(ORDER);
        const uint updates_idx = GET_UPDATES_INDEX(ORDER);
        INPUT2_TYPE val = updates[(int)updates_idx];
        INPUT1_TYPE index = indices[(int)indices_idx];
        if (index < 0) {index += SIZE;}
        ASSIGN_INDEX(index);
        const uint output_idx = GET_OUTPUT_INDEX(ORDER);
        #if HAS_FUSED_OPS
            FUSED_OPS_SECOND_KERNEL;
            output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
        #else
            output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
        #endif
    #endif
#elif ITER == 2 && REDUCE_MODE != 0
    uint ORDER;
    #if OUTPUT_DIMS == 4
        x = dim0;
        y = dim1;
        f = dim2 % OUTPUT_FEATURE_NUM;
        b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        x = dim0 % OUTPUT_SIZE_X;
        y = dim0 / OUTPUT_SIZE_X;
        z = dim1;
        f = dim2 % OUTPUT_FEATURE_NUM;
        b = dim2 / OUTPUT_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        x = dim0 % OUTPUT_SIZE_X;
        y = dim0 / OUTPUT_SIZE_X;
        z = dim1 % OUTPUT_SIZE_Z;
        w = dim1 / OUTPUT_SIZE_Z;
        f = dim2 % OUTPUT_FEATURE_NUM;
        b = dim2 / OUTPUT_FEATURE_NUM;
    #endif
    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);

    #if INPUT2_IS_FP == 1
        float val_f32 = FUNC_CALL(from_int)(output_fp[input_idx]);
        INPUT2_TYPE val = TO_OUTPUT_TYPE(val_f32);
    #else
        INPUT2_TYPE val = output_fp[input_idx];
    #endif

    #if REDUCE_MODE == MEAN_MODE
        if (count_k[output_idx] != 0) {
            val /= (count_k[output_idx] + USE_INIT_VAL);
        }
        #if USE_INIT_VAL == 0
        else {
            // Position was not scattered to, keep original data
            val = TO_OUTPUT_TYPE(data[input_idx]);
        }
        #endif
    #endif

    #if HAS_FUSED_OPS
        FUSED_OPS_SECOND_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
#endif
}

#ifdef REDUCE_MODE
    #undef SUM_MODE
    #undef PROD_MODE
    #undef MIN_MODE
    #undef MAX_MODE
    #undef MEAN_MODE
    #undef REDUCTION_NEUTRAL_VALUE
    #undef ATOMIC_REDUCE_OP
    #undef DEFINE_ATOMIC_REDUCE
    #undef CAS_OP
    #if INPUT2_IS_FP == 1 && INPUT2_TYPE_SIZE != 4
        #undef FP_SCALE
        #undef FP_SCALE_MAX
        #undef FP_SCALE_MIN
    #endif
    #undef FP_INT_MAX
    #undef FP_INT_MIN
    #undef FP_INT_ZERO
    #undef FP_INT_ONE
#endif

#undef GET_INDICES_INDEX
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef ORDER
#undef SIZE
#undef ASSIGN_INDEX
