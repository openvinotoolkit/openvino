// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define GET_INDICES_INDEX(idx_order) INPUT1_GET_INDEX(idx_order)
#define GET_UPDATES_INDEX(idx_order) INPUT2_GET_INDEX(idx_order)
#define GET_OUTPUT_INDEX(idx_order) OUTPUT_GET_INDEX(idx_order)
#if OUTPUT_DIMS == 4
    #define ORDER b,f,y,x
    #define IDX_ORDER idx_b,idx_f,idx_y,idx_x
#elif OUTPUT_DIMS == 5
    #define ORDER b,f,z,y,x
    #define IDX_ORDER idx_b,idx_f,idx_z,idx_y,idx_x
#elif OUTPUT_DIMS == 6
    #define ORDER b,f,w,z,y,x
    #define IDX_ORDER idx_b,idx_f,idx_w,idx_z,idx_y,idx_x
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

    #if USE_INIT_VAL == 0
        #if REDUCE_MODE == SUM_MODE
            #define REDUCTION_NEUTRAL_VALUE INPUT0_VAL_ZERO
        #elif REDUCE_MODE == PROD_MODE
            #define REDUCTION_NEUTRAL_VALUE INPUT0_VAL_ONE
        #elif REDUCE_MODE == MIN_MODE
            #define REDUCTION_NEUTRAL_VALUE INPUT0_VAL_MAX
        #elif REDUCE_MODE == MAX_MODE
            #define REDUCTION_NEUTRAL_VALUE INPUT0_VAL_MIN
        #elif REDUCE_MODE == MEAN_MODE
            #define REDUCTION_NEUTRAL_VALUE INPUT0_VAL_ZERO
        #else
            #error "Invalid REDUCE_MODE value"
        #endif
    #endif

    inline INPUT2_TYPE FUNC(reduce)(INPUT2_TYPE a, INPUT2_TYPE b)
    {
    #if REDUCE_MODE == SUM_MODE || REDUCE_MODE == MEAN_MODE
        return a + b;
    #elif REDUCE_MODE == PROD_MODE
        return a * b;
    #elif REDUCE_MODE == MIN_MODE
        return MIN(a, b);
    #elif REDUCE_MODE == MAX_MODE
        return MAX(a, b);
    #else
        #error "Invalid REDUCE_MODE value"
    #endif
    }

    #ifdef IS_SECOND_ITER // Socond kernel only
        #if REDUCE_MODE == MEAN_MODE
            inline void add_count(__local int count_k[], __local int count_v[], int idx, int count)
            {
                for (int i = 0; i < INPUT2_LENGTH; ++i) {
                    if (count_k[i] == -1) {
                        count_k[i] = idx;
                        count_v[i] = count;
                        break;
                    } else if (count_k[i] == idx) {
                        count_v[i] += count;
                        break;
                    }
                }
            }

            inline int get_count(__local int count_k[], __local int count_v[], int it, int *idx)
            {
                if (count_k[it] != -1) {
                    *idx = count_k[it];
                    count_k[it] = -1;
                    return count_v[it];
                }
                return -1;
            }
        #endif
    #endif
#endif

KERNEL(scatter_elements_update_ref)(const __global INPUT0_TYPE* data,
                   const __global INPUT1_TYPE* indices,
                   const __global INPUT2_TYPE* updates,
                   __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                   , FUSED_OPS_DECLS
#endif
)
{

    const uint dim0 = get_global_id(0);
    const uint dim1 = get_global_id(1);
    const uint dim2 = get_global_id(2);

#ifndef IS_SECOND_ITER // First kernel
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

    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[output_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif

#else // Second kernel
    #if OUTPUT_DIMS == 4
        const uint idx_x = dim0;
        const uint idx_y = dim1;
        const uint idx_f = dim2 % INPUT2_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT2_FEATURE_NUM;
    #elif OUTPUT_DIMS == 5
        const uint idx_x = dim0 % INPUT2_SIZE_X;
        const uint idx_y = dim0 / INPUT2_SIZE_X;
        const uint idx_z = dim1;
        const uint idx_f = dim2 % INPUT2_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT2_FEATURE_NUM;
    #elif OUTPUT_DIMS == 6
        const uint idx_x = dim0 % INPUT2_SIZE_X;
        const uint idx_y = dim0 / INPUT2_SIZE_X;
        const uint idx_z = dim1 % INPUT2_SIZE_Z;
        const uint idx_w = dim1 / INPUT2_SIZE_Z;
        const uint idx_f = dim2 % INPUT2_FEATURE_NUM;
        const uint idx_b = dim2 / INPUT2_FEATURE_NUM;
    #endif

    const uint indices_idx = GET_INDICES_INDEX(IDX_ORDER);
    INPUT1_TYPE index = indices[(int)indices_idx];

    #if OUTPUT_DIMS == 4
    #if     AXIS_VALUE == 0
        if (index < 0) { index += INPUT0_BATCH_NUM; }
        const uint x = idx_x; const uint y = idx_y; const uint f = idx_f; const uint b = index;
    #elif   AXIS_VALUE == 1
        if (index < 0) { index += INPUT0_FEATURE_NUM; }
        const uint x = idx_x; const uint y = idx_y; const uint f = index; const uint b = idx_b;
    #elif   AXIS_VALUE == 2
        if (index < 0) { index += INPUT0_SIZE_Y; }
        const uint x = idx_x; const uint y = index; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 3
        if (index < 0) { index += INPUT0_SIZE_X; }
        const uint x = index; const uint y = idx_y; const uint f = idx_f; const uint b = idx_b;
    #endif  // AXIS_VALUE
    #elif OUTPUT_DIMS == 5
    #if     AXIS_VALUE == 0
        if (index < 0) { index += INPUT0_BATCH_NUM; }
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint f = idx_f; const uint b = index;
    #elif   AXIS_VALUE == 1
        if (index < 0) { index += INPUT0_FEATURE_NUM; }
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint f = index; const uint b = idx_b;
    #elif   AXIS_VALUE == 2
        if (index < 0) { index += INPUT0_SIZE_Z; }
        const uint x = idx_x; const uint y = idx_y; const uint z = index; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 3
        if (index < 0) { index += INPUT0_SIZE_Y; }
        const uint x = idx_x; const uint y = index; const uint z = idx_z; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 4
        if (index < 0) { index += INPUT0_SIZE_X; }
        const uint x = index; const uint y = idx_y; const uint z = idx_z; const uint f = idx_f; const uint b = idx_b;
    #endif  // AXIS_VALUE
    #elif OUTPUT_DIMS == 6
    #if     AXIS_VALUE == 0
        if (index < 0) { index += INPUT0_BATCH_NUM; }
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint w = idx_w; const uint f = idx_f; const uint b = index;
    #elif   AXIS_VALUE == 1
        if (index < 0) { index += INPUT0_FEATURE_NUM; }
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint w = idx_w; const uint f = index; const uint b = idx_b;
    #elif   AXIS_VALUE == 2
        if (index < 0) { index += INPUT0_SIZE_W; }
        const uint x = idx_x; const uint y = idx_y; const uint z = idx_z; const uint w = index; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 3
        if (index < 0) { index += INPUT0_SIZE_Z; }
        const uint x = idx_x; const uint y = idx_y; const uint z = index; const uint w = idx_w; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 4
        if (index < 0) { index += INPUT0_SIZE_Y; }
        const uint x = idx_x; const uint y = index; const uint z = idx_z; const uint w = idx_w; const uint f = idx_f; const uint b = idx_b;
    #elif   AXIS_VALUE == 5
        if (index < 0) { index += INPUT0_SIZE_X; }
        const uint x = index; const uint y = idx_y; const uint z = idx_z; const uint w = idx_w; const uint f = idx_f; const uint b = idx_b;
    #endif  // AXIS_VALUE
    #endif
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);

    const uint updates_idx = GET_UPDATES_INDEX(IDX_ORDER);
    INPUT2_TYPE val = updates[(int)updates_idx];

    #ifdef REDUCE_MODE
        #if USE_INIT_VAL == 0
            output[output_idx] = REDUCTION_NEUTRAL_VALUE;
        #endif

        __local INPUT1_TYPE index_map[INPUT2_LENGTH];
        index_map[updates_idx] = output_idx;

        __local INPUT2_TYPE output_map[INPUT2_LENGTH];
        output_map[updates_idx] = output[output_idx];

        #if REDUCE_MODE == MEAN_MODE
            __local int count_k[INPUT2_LENGTH];
            __local int count_v[INPUT2_LENGTH];
        #endif

        barrier(CLK_LOCAL_MEM_FENCE);

        if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_local_id(2) == 0) {
            #if REDUCE_MODE == MEAN_MODE
                for (int i = 0; i < INPUT2_LENGTH; ++i) {
                    count_k[i] = -1;
                    count_v[i] = 0;
                }
            #endif

            for (int i = 0; i < INPUT2_LENGTH; ++i) {
                output[index_map[i]] = FUNC_CALL(reduce)(output[index_map[i]], updates[i]);

                #if REDUCE_MODE == MEAN_MODE
                    add_count(count_k, count_v, index_map[i], 1);
                #endif
            }

            #if REDUCE_MODE == MEAN_MODE
                for (int i = 0; i < INPUT2_LENGTH; ++i) {
                    int idx;
                    int count = get_count(count_k, count_v, i, &idx);
                    if (count != -1) {
                        output[idx] = output[idx] / (count + USE_INIT_VAL);
                    }
                }
            #endif
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        val = output[output_idx];
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
#endif

#undef GET_INDICES_INDEX
#undef GET_UPDATES_INDEX
#undef GET_OUTPUT_INDEX
#undef IDX_ORDER
#undef ORDER
