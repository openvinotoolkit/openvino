// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE ((OUTPUT_TYPE_SIZE * OUTPUT_FEATURE_PITCH) & 0xF == 0)

#if SUBGROUP_BLOCK_SIZE == 1
#define BLOCK_READ(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#define ACC_TYPE ACCUMULATOR_TYPE
#define TO_ACC_TYPE(x) TO_ACCUMULATOR_TYPE(x)
#define OUTPUT_VEC_TYPE OUTPUT_TYPE
#else
#define BLOCK_READ(ptr, offset) CAT(DT_INPUT_BLOCK_READ, SUBGROUP_BLOCK_SIZE)(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, SUBGROUP_BLOCK_SIZE)(ptr, offset, val)
#define ACC_TYPE MAKE_VECTOR_TYPE(ACCUMULATOR_TYPE, SUBGROUP_BLOCK_SIZE)
#define TO_ACC_TYPE(x) CAT(convert_, ACC_TYPE)(x)
#define OUTPUT_VEC_TYPE MAKE_VECTOR_TYPE(OUTPUT_TYPE, SUBGROUP_BLOCK_SIZE)
#endif

REQD_SUB_GROUP_SIZE(SUB_GROUP_SIZE)
KERNEL(rms_gpu_bfyx_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* gamma,
    __global OUTPUT_TYPE* output
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_DECLS
    #endif
)
{
    const uint data_idx = get_global_id(1);
    const uint in_data_idx = get_global_id(0);
    const uint workers_per_data = LWS;
    const uint data_size = DATA_SIZE;
    const uint items_num = data_size / workers_per_data;
    const uint leftovers = data_size % workers_per_data;

    const uint data_offset = data_idx * data_size;
    const uint subgroup_offset = get_sub_group_id() * get_sub_group_size() * items_num;

    ACCUMULATOR_TYPE data[STACK_SIZE];
    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;

    __local ACCUMULATOR_TYPE slm_buf[SLM_SIZE];

    uint i = 0;
    if (workers_per_data > SUB_GROUP_SIZE)
    {
        for (; i < items_num - (items_num % SUBGROUP_BLOCK_SIZE); i += SUBGROUP_BLOCK_SIZE)
        {
            ACC_TYPE vec_tmp = TO_ACC_TYPE(BLOCK_READ(input, data_offset + subgroup_offset + i * get_sub_group_size()));
#if SUBGROUP_BLOCK_SIZE == 1
            rms += native_powr(vec_tmp, 2);
            data[i] = vec_tmp;
#else
            unroll_for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++)
            {
                ACCUMULATOR_TYPE tmp = vec_tmp[j];
                rms += native_powr(tmp, 2);
                data[i + j] = tmp;
            }
#endif
        }
    }

    for (; i < items_num; i++)
    {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[data_offset + subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()]);
        rms += native_powr(tmp, 2);
        data[i] = tmp;
    }

    if (in_data_idx < leftovers)
    {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[data_offset + workers_per_data * items_num + in_data_idx]);
        rms += native_powr(tmp, 2);
        data[items_num] = tmp;
    }

    rms = sub_group_reduce_add(rms);

    if (get_sub_group_local_id() == 0)
        slm_buf[get_sub_group_id()] = rms;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint offset = get_num_sub_groups() / 2; offset > 0; offset /= 2) {
        if (in_data_idx < offset) {
            slm_buf[in_data_idx] += slm_buf[in_data_idx + offset];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (in_data_idx == 0) {
        rms = slm_buf[0] / data_size;
        slm_buf[0] = native_powr(sqrt(rms + TO_ACCUMULATOR_TYPE(EPSILON)), -1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    rms = slm_buf[0];

    #if HAS_FUSED_OPS
        uint b, f, z, y, x;
        #if INPUT_RANK == 1
            f = z = y = x = 1;
        #elif INPUT_RANK == 2
            z = y = x = 1;
            b = data_idx;
        #elif INPUT_RANK == 3
            x = 1;
            f = data_idx % OUTPUT_FEATURE_NUM;
            b = data_idx / OUTPUT_FEATURE_NUM;
        #else
            x = data_idx;
            y = x % OUTPUT_SIZE_Y;      x = x / OUTPUT_SIZE_Y;
            z = x % OUTPUT_SIZE_Z;      x = x / OUTPUT_SIZE_Z;
            f = x % OUTPUT_FEATURE_NUM; x = x / OUTPUT_FEATURE_NUM;
            b = x % OUTPUT_BATCH_NUM;   x = x / OUTPUT_BATCH_NUM;
        #endif
    #endif

    i = 0;
    if ((workers_per_data > SUB_GROUP_SIZE) && USE_BLOCK_WRITE)
    {
        for (; i < items_num - (items_num % SUBGROUP_BLOCK_SIZE); i += SUBGROUP_BLOCK_SIZE)
        {
            ACC_TYPE vec_gamma = TO_ACC_TYPE(BLOCK_READ(gamma, subgroup_offset + i * get_sub_group_size()));
            OUTPUT_VEC_TYPE vec_tmp;
            #if HAS_FUSED_OPS
                LAST_DIM = subgroup_offset + i * get_sub_group_size() + get_sub_group_local_id();
            #endif
#if SUBGROUP_BLOCK_SIZE == 1
            OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[i] * vec_gamma);
            #if HAS_FUSED_OPS
                FUSED_OPS;
                normalized = FUSED_OPS_RESULT;
            #endif
            vec_tmp = normalized;
#else
            unroll_for (int j = 0; j < SUBGROUP_BLOCK_SIZE; j++) {
                OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[i + j] * vec_gamma[j]);
                #if HAS_FUSED_OPS
                    LAST_DIM += j * get_sub_group_size();
                    FUSED_OPS;
                    normalized = FUSED_OPS_RESULT;
                #endif
                vec_tmp[j] = normalized;
            }
#endif
            BLOCK_WRITE(output, data_offset + subgroup_offset + i * get_sub_group_size(), vec_tmp);
        }
    }

    for (; i < items_num; i++)
    {
        ACCUMULATOR_TYPE temp = TO_ACCUMULATOR_TYPE(gamma[subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()]);
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[i] * temp);
        #if HAS_FUSED_OPS
            LAST_DIM = subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size();
            FUSED_OPS;
            normalized = FUSED_OPS_RESULT;
        #endif
        output[data_offset + subgroup_offset + get_sub_group_local_id() + i * get_sub_group_size()] = normalized;
    }

    if (in_data_idx < leftovers)
    {
        ACCUMULATOR_TYPE temp = TO_ACCUMULATOR_TYPE(gamma[workers_per_data * items_num + in_data_idx]);
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data[items_num] * temp);
        #if HAS_FUSED_OPS
            LAST_DIM = workers_per_data * items_num + in_data_idx;
            FUSED_OPS;
            normalized = FUSED_OPS_RESULT;
        #endif
        output[data_offset + workers_per_data * items_num + in_data_idx] = normalized;
    }
}
#undef USE_BLOCK_WRITE
#undef BLOCK_READ
#undef BLOCK_WRITE
#undef ACC_TYPE
#undef TO_ACC_TYPE
#undef OUTPUT_VEC_TYPE
