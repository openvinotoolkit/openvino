// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// SUM-reduction, dense-scatter fast path for ScatterElementsUpdate. See
// scatter_elements_update_kernel_opt_local_sum.h for the full design rationale and the
// measured evidence it's based on. This file intentionally mirrors
// scatter_elements_update_ref.cl's boilerplate (type macros, fixed-point to_int/from_int,
// atomic_reduce helpers) so the two kernels stay easy to compare -- only the ITER==1
// (update) body differs in substance; ITER==0/2 are the same computation as `_ref`'s
// SUM-mode path, just with the MEAN-mode/fused-ops/integer-type branches this kernel's
// narrow eligibility (see Validate()) never needs.

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

#if OUTPUT_DIMS != INPUT2_DIMS
    #error "OUTPUT_DIMS is supposed to be same as INPUT2_DIMS"
#endif

// This kernel is only ever selected for REDUCE_MODE == SUM (see Validate()); the
// accumulator encoding is the same scheme `_ref` uses, all three branches of it (see its
// own comments for the f32-bitcast-CAS / fp16-fixed-point-scale / integer-identity
// rationale). Kept identical so the two kernels' accumulators are bit-for-bit
// interchangeable for every element type either kernel accepts.
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

// f32 has no native OpenCL float atomics -- bit-reinterpret CAS, same as `_ref`. Note the
// CAS adds in *floating point*, so local staging reassociates an f32 sum; `_ref`'s own
// unordered global atomics already leave that order unspecified (see the header). Both
// other encodings are plain int32, so they use a real hardware atomic_fetch_add at both
// local and global scope, and their sums reassociate exactly.
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
    // `scope` is honoured and the ordering is relaxed deliberately. This kernel performs a
    // pure accumulation: only the atomicity of each add is required, not ordering against
    // other memory operations. The ordering that is required comes from the barriers, which
    // sequence the local staging window against its flush, and from the kernel boundary
    // between the accumulate and finalize stages.
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

KERNEL(scatter_elements_update_opt_local_sum)(OPTIONAL_SHAPE_INFO_ARG
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

    // Anchor the window on work-item 0's own destination -- a cheap, no-extra-pass
    // locality guess. Threads whose real destination lands elsewhere in the window
    // still benefit; threads outside it fall back below, so a bad guess only costs
    // effectiveness, never correctness.
    __local int window_base_local;
    if (lid == 0) {
        window_base_local = (int)output_idx;
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

    // Flush only touched (nonzero) slots -- skipping a net-zero slot is exact, not an
    // approximation: adding zero never changes the sum, regardless of whether it's
    // genuinely untouched or the sum of contributions that happened to cancel out.
    for (uint i = lid; i < WINDOW_SIZE; i += lsize) {
        int v = local_window[i];
        if (v != FP_INT_ZERO) {
            long gidx = (long)window_base + (long)i;
            if (gidx >= 0 && gidx < OPT_LOCAL_ACC_TOTAL_ELEMENTS) {
                FUNC_CALL(atomic_add_global)(&output_fp[gidx], v);
            }
        }
    }

#elif ITER == 2  // Finalize: decode the fixed-point accumulator back to the real output type.
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
#undef ORDER
#undef SIZE
#undef ASSIGN_INDEX
#undef FP_SCALE
#undef FP_SCALE_MAX
#undef FP_SCALE_MIN
#undef FP_INT_ZERO
#undef CAS_ADD
#undef ATOMIC_ADD_OP
