// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// SUM-reduction fast path for ScatterElementsUpdate. The boilerplate mirrors
// scatter_elements_update_ref.cl; only the ITER == 1 body differs in substance.

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
#endif

// Accumulator encoding, identical to _ref's, so the two are interchangeable.
#define FP_SCALE     65504.0f
#define FP_SCALE_MAX 2147483648.0f
#define FP_SCALE_MIN -FP_SCALE_MAX
#define FP_INT_ZERO 0

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
    #if INPUT2_TYPE_SIZE == 4
        return as_float(acc);
    #else
        return convert_float(acc) / FP_SCALE;
    #endif
}

// f32 has no native OpenCL float atomics -- bit-reinterpret CAS, same as _ref.
#if INPUT2_IS_FP && INPUT2_TYPE_SIZE == 4
    #define CAS_ADD(addr, val, scope) { \
        int expected_value; \
        int desired_value; \
        bool success; \
        do { \
            expected_value = atomic_load_explicit(addr, memory_order_acquire, scope); \
            desired_value  = as_int(as_float(expected_value) + as_float(val)); \
            success = atomic_compare_exchange_weak_explicit(addr, &expected_value, desired_value, \
                          memory_order_acq_rel, memory_order_acquire, scope); \
        } while (!success); \
    }
    #define ATOMIC_ADD_OP(addr, val, scope) CAS_ADD(addr, val, scope)
#else
    // Pure accumulation: only per-add atomicity is needed. The ordering comes from the
    // barriers below and from the kernel boundary before the finalize stage.
    #define ATOMIC_ADD_OP(addr, val, scope) \
        atomic_fetch_add_explicit(addr, val, memory_order_relaxed, scope)
#endif

inline void FUNC(atomic_add_local)(volatile __local int *ptr, int val)
{
    atomic_int *atomic_addr = (atomic_int *)ptr;
    ATOMIC_ADD_OP(atomic_addr, val, memory_scope_work_group);
}

inline void FUNC(atomic_add_global)(volatile __global int *ptr, int val)
{
    atomic_int *atomic_addr = (atomic_int *)ptr;
    ATOMIC_ADD_OP(atomic_addr, val, memory_scope_device);
}

KERNEL(scatter_elements_update_opt_local_sum)(
                   OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates,
                   __global OUTPUT_TYPE* output,
                   __global int* output_fp
#if ITER == 1
                   , __local int* local_window
#endif
)
{
    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

#if ITER == 0  // Initialization: seed the fixed-point accumulator from `data`.
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
    #endif
    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    output_fp[output_idx] = FUNC_CALL(to_int)(data[input_idx]);

#elif ITER == 1  // Update: local-staged atomic accumulate, global fallback outside the window.
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
    #endif
    const uint indices_idx = GET_INDICES_INDEX(ORDER);
    const uint updates_idx = GET_UPDATES_INDEX(ORDER);
    INPUT2_TYPE val = updates[(int)updates_idx];
    INPUT1_TYPE index = indices[(int)indices_idx];
    if (index < 0) { index += SIZE; }
    ASSIGN_INDEX(index);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    const int val_fixed = FUNC_CALL(to_int)(val);

    const uint lsize = get_local_size(0) * get_local_size(1) * get_local_size(2);
    const uint lid = get_local_id(0) + get_local_size(0) * (get_local_id(1) + get_local_size(1) * get_local_id(2));

    // WINDOW_ANCHOR_ZERO means the whole accumulator fits the window, so every destination
    // is in range. Otherwise guess locality from work-item 0's destination; misses take the
    // global atomic below, so a bad guess costs effectiveness, never correctness.
    __local int window_base_local;
    if (lid == 0) {
#if WINDOW_ANCHOR_ZERO
        window_base_local = 0;
#else
        window_base_local = (int)output_idx;
#endif
    }

    for (uint i = lid; i < WINDOW_SIZE; i += lsize) {
        local_window[i] = FP_INT_ZERO;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int window_base = window_base_local;
    const long rel = (long)output_idx - (long)window_base;

    if (rel >= 0 && rel < WINDOW_SIZE) {
        FUNC_CALL(atomic_add_local)(&local_window[rel], val_fixed);
    } else {
        FUNC_CALL(atomic_add_global)(&output_fp[output_idx], val_fixed);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Flush only touched slots; skipping a net-zero one is exact. The accumulator carries
    // WINDOW_SIZE of slack, so a window anchored near the end cannot run past it.
    for (uint i = lid; i < WINDOW_SIZE; i += lsize) {
        int v = local_window[i];
        if (v != FP_INT_ZERO) {
            FUNC_CALL(atomic_add_global)(&output_fp[window_base + i], v);
        }
    }

#elif ITER == 2  // Finalize: decode the fixed-point accumulator back to the output type.
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
    #endif
    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    #if INPUT2_IS_FP
        float val_f32 = FUNC_CALL(from_int)(output_fp[input_idx]);
        INPUT2_TYPE val = TO_OUTPUT_TYPE(val_f32);
    #else
        INPUT2_TYPE val = output_fp[input_idx];
    #endif
    output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
#endif
}

#undef GET_INDICES_INDEX
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef GET_INPUT_INDEX
#undef ORDER
#undef SIZE
#undef ASSIGN_INDEX
#undef FP_SCALE
#undef FP_SCALE_MAX
#undef FP_SCALE_MIN
#undef FP_INT_ZERO
#undef CAS_ADD
#undef ATOMIC_ADD_OP
