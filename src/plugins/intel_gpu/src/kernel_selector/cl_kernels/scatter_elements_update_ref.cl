// Copyright (C) 2018-2025 Intel Corporation
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

#ifdef IS_SECOND_ITER // Socond kernel only
    #ifdef REDUCE_MODE
        #define COUNT_LIMIT 4096
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
        #if REDUCE_MODE == SUM_MODE
            return a + b;
        #elif REDUCE_MODE == PROD_MODE
            return a * b;
        #elif REDUCE_MODE == MIN_MODE
            return MIN(a, b);
        #elif REDUCE_MODE == MAX_MODE
            return MAX(a, b);
        #elif REDUCE_MODE == MEAN_MODE
            return a + b;
        #else
            #error "Invalid REDUCE_MODE value"
        #endif
        }

        inline uint add_count(__local int count_k[], __local int count_v[], int idx, uint valid_count)
        {
            for (int i = 0; i < valid_count; ++i) {
                if (count_k[i] == idx) {
                    count_v[i] += 1;
                    return valid_count;
                }
            }
            count_k[valid_count] = idx;
            count_v[valid_count] += 1;
            return valid_count + 1;
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

KERNEL(scatter_elements_update_ref)(OPTIONAL_SHAPE_INFO_ARG
                   const __global INPUT0_TYPE* data,
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
    const uint input_idx = GET_INPUT_INDEX(ORDER);
    const uint output_idx = GET_OUTPUT_INDEX(ORDER);
    INPUT0_TYPE val = data[input_idx];
    #if HAS_FUSED_OPS
        FUSED_OPS_FIRST_KERNEL;
        output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_FIRST_KERNEL);
    #else
        output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
    #endif
#else
    #ifdef REDUCE_MODE
        #if OUTPUT_DIMS == 4
            const uint tgx = INPUT2_SIZE_X;
            const uint tgy = INPUT2_SIZE_Y;
        #elif OUTPUT_DIMS == 5
            const uint tgx = INPUT2_SIZE_X * INPUT2_SIZE_Y;
            const uint tgy = INPUT2_SIZE_Z;
        #elif OUTPUT_DIMS == 6
            const uint tgx = INPUT2_SIZE_X * INPUT2_SIZE_Y;
            const uint tgy = INPUT2_SIZE_Z * INPUT2_SIZE_W;
        #endif
        const uint tgz = INPUT2_FEATURE_NUM * INPUT2_BATCH_NUM;
        #if INPUT2_LENGTH == 0 || INPUT2_LENGTH > COUNT_LIMIT
            #define COUNT_LENGTH COUNT_LIMIT   // Maximum number of elements to reduce in case of shape agnostic kernel or large shapes
        #else
            #define COUNT_LENGTH INPUT2_LENGTH
        #endif
        __local int count_k[COUNT_LENGTH];
        __local int count_v[COUNT_LENGTH];
        for (int i = 0; i < COUNT_LENGTH; ++i) {
            count_k[i] = -1;
            count_v[i] = 0;
        }
        const uint input2_length = tgx * tgy * tgz > COUNT_LENGTH ? COUNT_LENGTH : tgx * tgy * tgz;
        #if USE_INIT_VAL == 0
            for (uint gz = 0; gz < tgz; gz++) {
                for (uint gy = 0; gy < tgy; gy++) {
                    for (uint gx = 0; gx < tgx; gx++) {
                        uint ORDER;
                        #if OUTPUT_DIMS == 4
                            x = gx;
                            y = gy;
                        #elif OUTPUT_DIMS == 5
                            x = gx % INPUT2_SIZE_X;
                            y = gx / INPUT2_SIZE_X;
                            z = gy;
                        #elif OUTPUT_DIMS == 6
                            x = gx % INPUT2_SIZE_X;
                            y = gx / INPUT2_SIZE_X;
                            z = gy % INPUT2_SIZE_Z;
                            w = gy / INPUT2_SIZE_Z;
                        #endif
                        f = gz % INPUT2_FEATURE_NUM;
                        b = gz / INPUT2_FEATURE_NUM;
                        const uint indices_idx = GET_INDICES_INDEX(ORDER);
                        INPUT1_TYPE index = indices[(int)indices_idx];
                        if (index < 0) { index += SIZE; }
                        ASSIGN_INDEX(index);
                        const uint output_idx = GET_OUTPUT_INDEX(ORDER);
                        output[output_idx] = REDUCTION_NEUTRAL_VALUE;
                    }
                }
            }
        #endif
        uint valid_count = 0;
        for (uint gz = 0; gz < tgz; gz++) {
            for (uint gy = 0; gy < tgy; gy++) {
                for (uint gx = 0; gx < tgx; gx++) {
                    uint ORDER;
                    #if OUTPUT_DIMS == 4
                        x = gx;
                        y = gy;
                    #elif OUTPUT_DIMS == 5
                        x = gx % INPUT2_SIZE_X;
                        y = gx / INPUT2_SIZE_X;
                        z = gy;
                    #elif OUTPUT_DIMS == 6
                        x = gx % INPUT2_SIZE_X;
                        y = gx / INPUT2_SIZE_X;
                        z = gy % INPUT2_SIZE_Z;
                        w = gy / INPUT2_SIZE_Z;
                    #endif
                    f = gz % INPUT2_FEATURE_NUM;
                    b = gz / INPUT2_FEATURE_NUM;
                     const uint indices_idx = GET_INDICES_INDEX(ORDER);
                     const uint updates_idx = GET_UPDATES_INDEX(ORDER);
                     INPUT2_TYPE val = updates[(int)updates_idx];
                     INPUT1_TYPE index = indices[(int)indices_idx];
                     if (index < 0) {index += SIZE;}
                     ASSIGN_INDEX(index);
                     const uint output_idx = GET_OUTPUT_INDEX(ORDER);
                     val = FUNC_CALL(reduce)(output[output_idx], val);
                     output[output_idx] = val;
                     if (valid_count < COUNT_LENGTH) {
                        valid_count = add_count(count_k, count_v, output_idx, valid_count);
                     } else {
                        printf("Error: scatter_elements_update_ref on unexpected shape.\n");
                     }
                 }
             }
         }
        for (int i = 0; i < valid_count; ++i) {
            int output_idx;
            const int count = get_count(count_k, count_v, i, &output_idx);
            #if REDUCE_MODE==MEAN_MODE
                output[output_idx] = output[output_idx] / (count + USE_INIT_VAL);
            #endif
            INPUT2_TYPE val = output[output_idx];
            #if HAS_FUSED_OPS
                FUSED_OPS_SECOND_KERNEL;
                output[output_idx] = TO_OUTPUT_TYPE(FUSED_OPS_RESULT_SECOND_KERNEL);
            #else
                output[output_idx] = ACTIVATION(val, ACTIVATION_PARAMS);
            #endif
        }
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
#undef ORDER
#undef SIZE
#undef ASSIGN_INDEX
