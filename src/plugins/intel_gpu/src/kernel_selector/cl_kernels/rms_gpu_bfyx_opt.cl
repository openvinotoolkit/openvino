// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

// Check alignment restrictions for using block writes on output.
#define USE_BLOCK_WRITE (((OUTPUT_TYPE_SIZE * OUTPUT_FEATURE_PITCH) & 0xF) == 0)

// Generalized RMS optimization:
// - The host selector chooses LWS as the largest power of two that keeps at least 8
//   normalized elements per work-item, clamped to SUB_GROUP_SIZE. This gives the measured
//   roofline choices for Qwen3-Omni C6 without hardcoding shapes (`D=128 -> LWS=16`,
//   `D=2560 -> LWS=256`).
// - `ONE_SUBGROUP_ROW` is emitted when one subgroup covers a row. It removes SLM and all
//   barriers because `sub_group_reduce_add` already has the whole row reduction.
// - `MULTI_SUBGROUP_ROW` is emitted when multiple subgroups cover a row. Each subgroup
//   writes one partial sum to SLM, then every subgroup redundantly reduces those partials,
//   so only one barrier is needed and no broadcast barrier is required.
// - `STACK_SIZE` is `ceil(DATA_SIZE / LWS)`. The fast path caches input in registers and
//   reads it once; `RMS_REREAD_INPUT=1` skips that register array for larger stacks and
//   rereads input during the output pass to avoid excessive register pressure.
#ifndef ONE_SUBGROUP_ROW
#define ONE_SUBGROUP_ROW 0
#endif

#ifndef MULTI_SUBGROUP_ROW
#define MULTI_SUBGROUP_ROW 0
#endif

#ifndef RMS_REREAD_INPUT
#define RMS_REREAD_INPUT 0
#endif

#ifndef HAS_FUSED_OPS
#define HAS_FUSED_OPS 0
#endif

#ifndef HAS_FUSED_OPS_DECLS
#define HAS_FUSED_OPS_DECLS 0
#endif

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
#if ELEMENTWISE_AFFINE
    const __global INPUT1_TYPE* gamma,
#endif
    __global OUTPUT_TYPE* output
    #if HAS_FUSED_OPS_DECLS
        , FUSED_OPS_DECLS
    #endif
)
{
    const uint data_idx = get_global_id(1);
    const uint in_data_idx = get_global_id(0);
    const uint local_data_idx = get_local_id(0);
    const uint workers_per_data = LWS;
    const uint data_size = DATA_SIZE;
    const uint items_num = data_size / workers_per_data;
    const uint leftovers = data_size % workers_per_data;

    #if HAS_PADDING
        uint b_idx = 0;
        uint f_idx = 0;
        uint z_idx = 0;
        uint y_idx = 0;
        uint x_idx = 0;
        #if INPUT_RANK == 2
            b_idx = (data_idx);
        #elif INPUT_RANK == 3
            f_idx = (data_idx % (INPUT0_FEATURE_NUM));
            b_idx = (data_idx / (INPUT0_FEATURE_NUM));
        #else
            y_idx = (data_idx % (INPUT0_SIZE_Y));
            z_idx = (data_idx / (INPUT0_SIZE_Y)) % INPUT0_SIZE_Z;
            f_idx = (data_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_Z)) % INPUT0_FEATURE_NUM;
            b_idx = (data_idx / (INPUT0_SIZE_Y * INPUT0_SIZE_Z * INPUT0_FEATURE_NUM)) % INPUT0_BATCH_NUM;
        #endif

        const uint input_data_offset = FUNC_CALL(get_input_index)(OPTIONAL_SHAPE_INFO_TENSOR b_idx, f_idx, 0, z_idx, y_idx, x_idx);
    #else
        const uint input_data_offset = data_idx * data_size;
    #endif

    const uint output_data_offset = data_idx * data_size;

    const uint sgs = get_sub_group_size();
    const uint subgroup_offset = get_sub_group_id() * sgs * items_num;

#if !RMS_REREAD_INPUT
    ACCUMULATOR_TYPE data[STACK_SIZE];
#endif
    ACCUMULATOR_TYPE rms = ACCUMULATOR_VAL_ZERO;

#if !ONE_SUBGROUP_ROW
    __local ACCUMULATOR_TYPE slm_buf[SLM_SIZE];
#endif

    uint i = 0;
    if (workers_per_data >= SUB_GROUP_SIZE)
    {
        const uint ibase = input_data_offset + subgroup_offset;
        for (; i + 8 <= items_num; i += 8) {
            MAKE_VECTOR_TYPE(INPUT0_TYPE, 8) v = DT_INPUT_BLOCK_READ8(input, ibase + i * sgs);
            unroll_for (int j = 0; j < 8; j++) {
                ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(v[j]);
                rms += tmp * tmp;
#if !RMS_REREAD_INPUT
                data[i + j] = tmp;
#endif
            }
        }
        for (; i + 4 <= items_num; i += 4) {
            MAKE_VECTOR_TYPE(INPUT0_TYPE, 4) v = DT_INPUT_BLOCK_READ4(input, ibase + i * sgs);
            unroll_for (int j = 0; j < 4; j++) {
                ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(v[j]);
                rms += tmp * tmp;
#if !RMS_REREAD_INPUT
                data[i + j] = tmp;
#endif
            }
        }
        for (; i + 2 <= items_num; i += 2) {
            MAKE_VECTOR_TYPE(INPUT0_TYPE, 2) v = DT_INPUT_BLOCK_READ2(input, ibase + i * sgs);
            unroll_for (int j = 0; j < 2; j++) {
                ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(v[j]);
                rms += tmp * tmp;
#if !RMS_REREAD_INPUT
                data[i + j] = tmp;
#endif
            }
        }
    }

    for (; i < items_num; i++)
    {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[input_data_offset + subgroup_offset + get_sub_group_local_id() + i * sgs]);
        rms += tmp * tmp;
#if !RMS_REREAD_INPUT
        data[i] = tmp;
#endif
    }

    if (leftovers != 0 && local_data_idx < leftovers)
    {
        ACCUMULATOR_TYPE tmp = TO_ACCUMULATOR_TYPE(input[input_data_offset + workers_per_data * items_num + local_data_idx]);
        rms += tmp * tmp;
#if !RMS_REREAD_INPUT
        data[items_num] = tmp;
#endif
    }

    rms = sub_group_reduce_add(rms);

#if ONE_SUBGROUP_ROW
    rms = native_rsqrt(rms / data_size + TO_ACCUMULATOR_TYPE(EPSILON));
#elif MULTI_SUBGROUP_ROW
    if (get_sub_group_local_id() == 0)
        slm_buf[get_sub_group_id()] = rms;

    barrier(CLK_LOCAL_MEM_FENCE);
    const uint nsg = LWS / SUB_GROUP_SIZE;
    ACCUMULATOR_TYPE p = ACCUMULATOR_VAL_ZERO;
    for (uint k = get_sub_group_local_id(); k < nsg; k += sgs)
        p += slm_buf[k];
    p = sub_group_reduce_add(p);
    rms = native_rsqrt(p / data_size + TO_ACCUMULATOR_TYPE(EPSILON));
#else
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
#endif

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
    if ((workers_per_data > SUB_GROUP_SIZE) && USE_BLOCK_WRITE && !HAS_FUSED_OPS)
    {
        const uint obase = output_data_offset + subgroup_offset;
        const uint gbase = subgroup_offset;
        for (; i + 8 <= items_num; i += 8) {
#if ELEMENTWISE_AFFINE
            MAKE_VECTOR_TYPE(INPUT1_TYPE, 8) g = DT_INPUT_BLOCK_READ8(gamma, gbase + i * sgs);
#endif
            MAKE_VECTOR_TYPE(OUTPUT_TYPE, 8) o;
            unroll_for (int j = 0; j < 8; j++) {
#if RMS_REREAD_INPUT
                ACCUMULATOR_TYPE data_value = TO_ACCUMULATOR_TYPE(input[input_data_offset + subgroup_offset + get_sub_group_local_id() + (i + j) * sgs]);
#else
                ACCUMULATOR_TYPE data_value = data[i + j];
#endif
#if ELEMENTWISE_AFFINE
                o[j] = TO_OUTPUT_TYPE(rms * data_value * TO_ACCUMULATOR_TYPE(g[j]));
#else
                o[j] = TO_OUTPUT_TYPE(rms * data_value);
#endif
            }
            DT_OUTPUT_BLOCK_WRITE8(output, obase + i * sgs, o);
        }
        for (; i + 4 <= items_num; i += 4) {
#if ELEMENTWISE_AFFINE
            MAKE_VECTOR_TYPE(INPUT1_TYPE, 4) g = DT_INPUT_BLOCK_READ4(gamma, gbase + i * sgs);
#endif
            MAKE_VECTOR_TYPE(OUTPUT_TYPE, 4) o;
            unroll_for (int j = 0; j < 4; j++) {
#if RMS_REREAD_INPUT
                ACCUMULATOR_TYPE data_value = TO_ACCUMULATOR_TYPE(input[input_data_offset + subgroup_offset + get_sub_group_local_id() + (i + j) * sgs]);
#else
                ACCUMULATOR_TYPE data_value = data[i + j];
#endif
#if ELEMENTWISE_AFFINE
                o[j] = TO_OUTPUT_TYPE(rms * data_value * TO_ACCUMULATOR_TYPE(g[j]));
#else
                o[j] = TO_OUTPUT_TYPE(rms * data_value);
#endif
            }
            DT_OUTPUT_BLOCK_WRITE4(output, obase + i * sgs, o);
        }
        for (; i + 2 <= items_num; i += 2) {
#if ELEMENTWISE_AFFINE
            MAKE_VECTOR_TYPE(INPUT1_TYPE, 2) g = DT_INPUT_BLOCK_READ2(gamma, gbase + i * sgs);
#endif
            MAKE_VECTOR_TYPE(OUTPUT_TYPE, 2) o;
            unroll_for (int j = 0; j < 2; j++) {
#if RMS_REREAD_INPUT
                ACCUMULATOR_TYPE data_value = TO_ACCUMULATOR_TYPE(input[input_data_offset + subgroup_offset + get_sub_group_local_id() + (i + j) * sgs]);
#else
                ACCUMULATOR_TYPE data_value = data[i + j];
#endif
#if ELEMENTWISE_AFFINE
                o[j] = TO_OUTPUT_TYPE(rms * data_value * TO_ACCUMULATOR_TYPE(g[j]));
#else
                o[j] = TO_OUTPUT_TYPE(rms * data_value);
#endif
            }
            DT_OUTPUT_BLOCK_WRITE2(output, obase + i * sgs, o);
        }
    }

    for (; i < items_num; i++)
    {
#if RMS_REREAD_INPUT
        ACCUMULATOR_TYPE data_value = TO_ACCUMULATOR_TYPE(input[input_data_offset + subgroup_offset + get_sub_group_local_id() + i * sgs]);
#else
        ACCUMULATOR_TYPE data_value = data[i];
#endif
#if ELEMENTWISE_AFFINE
        ACCUMULATOR_TYPE temp = TO_ACCUMULATOR_TYPE(gamma[subgroup_offset + get_sub_group_local_id() + i * sgs]);
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data_value * temp);
#else
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data_value);
#endif
        #if HAS_FUSED_OPS
            LAST_DIM = subgroup_offset + get_sub_group_local_id() + i * sgs;
            FUSED_OPS;
            normalized = FUSED_OPS_RESULT;
        #endif
        output[output_data_offset + subgroup_offset + get_sub_group_local_id() + i * sgs] = normalized;
    }

    if (leftovers != 0 && local_data_idx < leftovers)
    {
#if RMS_REREAD_INPUT
        ACCUMULATOR_TYPE data_value = TO_ACCUMULATOR_TYPE(input[input_data_offset + workers_per_data * items_num + local_data_idx]);
#else
        ACCUMULATOR_TYPE data_value = data[items_num];
#endif
#if ELEMENTWISE_AFFINE
        ACCUMULATOR_TYPE temp = TO_ACCUMULATOR_TYPE(gamma[workers_per_data * items_num + local_data_idx]);
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data_value * temp);
#else
        OUTPUT_TYPE normalized = TO_OUTPUT_TYPE(rms * data_value);
#endif
        #if HAS_FUSED_OPS
            LAST_DIM = workers_per_data * items_num + local_data_idx;
            FUSED_OPS;
            normalized = FUSED_OPS_RESULT;
        #endif
        output[output_data_offset + workers_per_data * items_num + local_data_idx] = normalized;
    }
}
#undef USE_BLOCK_WRITE
#undef BLOCK_READ
#undef BLOCK_WRITE
#undef ACC_TYPE
#undef TO_ACC_TYPE
#undef OUTPUT_VEC_TYPE
