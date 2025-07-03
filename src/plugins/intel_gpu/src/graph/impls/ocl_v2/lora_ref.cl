// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#ifdef BASE_KERNEL
KERNEL(lora_ref)(OPTIONAL_SHAPE_INFO_ARG
                 const __global OUTPUT_TYPE* main_input,
                 const __global INPUT1_TYPE* lora_input,
                 const __global STATE_TYPE* state_a,
                 const __global STATE_TYPE* state_alpha,
                 const __global STATE_TYPE* state_b,
                       __global OUTPUT_TYPE* output)
{
    const uint K = INPUT1_SIZE_Y;

    uint bf = get_group_id(0);
    uint yx = get_group_id(1);

    uint local_id = get_local_id(1);

    ACCUMULATOR_TYPE acc = ACCUMULATOR_VAL_ZERO;

    for (uint ki = 0; ki < K; ++ki) {
        uint lora_input_idx = bf * K + ki;
        uint state_a_idx = ki * LORA_RANK + local_id;

        acc = mad(TO_ACCUMULATOR_TYPE(lora_input[lora_input_idx]),
                  TO_ACCUMULATOR_TYPE(state_a[state_a_idx]),
                  acc);
    }

    acc *= TO_ACCUMULATOR_TYPE(state_alpha[local_id]) / TO_ACCUMULATOR_TYPE(LORA_RANK);

    __local ACCUMULATOR_TYPE tmp_buf[MAX_LORA_RANK];
    tmp_buf[local_id] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    ACCUMULATOR_TYPE final_acc = ACCUMULATOR_VAL_ZERO;
    uint new_yx = yx * LORA_RANK + local_id;

    if (new_yx >= INPUT0_SIZE_Y * INPUT0_SIZE_X) {
        return;
    }

    for (uint ki = 0; ki < LORA_RANK; ++ki) {
        uint state_b_idx = ki * INPUT0_SIZE_Y * INPUT0_SIZE_X + new_yx;

        final_acc = mad(tmp_buf[ki], TO_ACCUMULATOR_TYPE(state_b[state_b_idx]), final_acc);
    }
    uint output_idx = bf * INPUT0_SIZE_Y * INPUT0_SIZE_X + new_yx;
    output[output_idx] = TO_OUTPUT_TYPE(final_acc) + main_input[output_idx];
}
#endif

#ifdef HORIZONTAL_FUSED
KERNEL(lora_ref)(OPTIONAL_SHAPE_INFO_ARG
                 const __global OUTPUT_TYPE* main_input,
                 const __global INPUT1_TYPE* lora_input,
                 const __global STATE_TYPE* state_a_0,
                 const __global STATE_TYPE* state_alpha_0,
                 const __global STATE_TYPE* state_b_0,
                 const __global STATE_TYPE* state_a_1,
                 const __global STATE_TYPE* state_alpha_1,
                 const __global STATE_TYPE* state_b_1,
#if LORA_COUNT == 3
                 const __global STATE_TYPE* state_a_2,
                 const __global STATE_TYPE* state_alpha_2,
                 const __global STATE_TYPE* state_b_2,
#endif
                       __global OUTPUT_TYPE* output)
{
    const uint K = INPUT1_SIZE_Y;

    const uint bf = get_group_id(0);
    const uint yx = get_group_id(1);
    const uint local_id = get_local_id(1);

    ACCUMULATOR_TYPE acc0 = ACCUMULATOR_VAL_ZERO;
    ACCUMULATOR_TYPE acc1 = ACCUMULATOR_VAL_ZERO;
#if LORA_COUNT == 3
    ACCUMULATOR_TYPE acc2 = ACCUMULATOR_VAL_ZERO;
#endif

    for (uint ki = 0; ki < K; ++ki) {
        uint lora_idx = bf * K + ki;
        uint state_idx = ki * LORA_RANK + local_id;

        ACCUMULATOR_TYPE lora_val = TO_ACCUMULATOR_TYPE(lora_input[lora_idx]);

        acc0 = mad(lora_val, TO_ACCUMULATOR_TYPE(state_a_0[state_idx]), acc0);
        acc1 = mad(lora_val, TO_ACCUMULATOR_TYPE(state_a_1[state_idx]), acc1);
#if LORA_COUNT == 3
        acc2 = mad(lora_val, TO_ACCUMULATOR_TYPE(state_a_2[state_idx]), acc2);
#endif
    }

    acc0 *= TO_ACCUMULATOR_TYPE(state_alpha_0[local_id]) / TO_ACCUMULATOR_TYPE(LORA_RANK);
    acc1 *= TO_ACCUMULATOR_TYPE(state_alpha_1[local_id]) / TO_ACCUMULATOR_TYPE(LORA_RANK);
#if LORA_COUNT == 3
    acc2 *= TO_ACCUMULATOR_TYPE(state_alpha_2[local_id]) / TO_ACCUMULATOR_TYPE(LORA_RANK);
#endif

    __local ACCUMULATOR_TYPE tmp_buf[LORA_COUNT * MAX_LORA_RANK];
    tmp_buf[local_id] = acc0;
    tmp_buf[local_id + LORA_RANK] = acc1;
#if LORA_COUNT == 3
    tmp_buf[local_id + 2 * LORA_RANK] = acc2;
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint range_0_end = INPUT4_BATCH_NUM;
    const uint range_1_end = range_0_end + INPUT7_BATCH_NUM;
#if LORA_COUNT == 3
    const uint range_2_end = range_1_end + INPUT10_BATCH_NUM;
#endif

    const uint proc_range_start[LORA_COUNT] = {
        0, range_0_end
#if LORA_COUNT == 3
        , range_1_end
#endif
    };

    const uint proc_range_end[LORA_COUNT] = {
        range_0_end, range_1_end
#if LORA_COUNT == 3
        , range_2_end
#endif
    };

    const __global STATE_TYPE* state_b[LORA_COUNT] = {
        state_b_0, state_b_1
#if LORA_COUNT == 3
        , state_b_2
#endif
    };

    __attribute__((opencl_unroll_hint(LORA_COUNT)))
    for (uint i = 0; i < LORA_COUNT; ++i) {
        uint tmp_offset = i * LORA_RANK;

        uint processing_size = proc_range_end[i] - proc_range_start[i];
        uint num_blocks = processing_size / LORA_RANK;
        uint leftover = processing_size % LORA_RANK;

        for (uint block = 0; block < num_blocks; ++block) {
            uint yx_range = proc_range_start[i] + block * LORA_RANK + local_id;

            ACCUMULATOR_TYPE final_acc = ACCUMULATOR_VAL_ZERO;

            for (uint ki = 0; ki < LORA_RANK; ++ki) {
                uint state_b_idx = ki * processing_size + (yx_range - proc_range_start[i]);

                final_acc = mad(tmp_buf[tmp_offset + ki],
                                TO_ACCUMULATOR_TYPE(state_b[i][state_b_idx]),
                                final_acc);
            }
            uint output_idx = bf * INPUT0_SIZE_Y * INPUT0_SIZE_X + yx_range;
            output[output_idx] = TO_OUTPUT_TYPE(final_acc) + main_input[output_idx];
        }

        if (local_id < leftover) {
            uint yx_range = proc_range_start[i] + num_blocks * LORA_RANK + local_id;

            ACCUMULATOR_TYPE final_acc = ACCUMULATOR_VAL_ZERO;

            for (uint ki = 0; ki < LORA_RANK; ++ki) {
                uint state_b_idx = ki * processing_size + (yx_range - proc_range_start[i]);

                final_acc = mad(tmp_buf[tmp_offset + ki],
                                TO_ACCUMULATOR_TYPE(state_b[i][state_b_idx]),
                                final_acc);
            }
            uint output_idx = bf * INPUT0_SIZE_Y * INPUT0_SIZE_X + yx_range;
            output[output_idx] = TO_OUTPUT_TYPE(final_acc) + main_input[output_idx];
        }
    }
}
#endif

#ifdef FUSED_OPS_KERNEL
KERNEL(fused_ops)(OPTIONAL_SHAPE_INFO_ARG
                  __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
                , FUSED_OPS_DECLS
#endif
                    )
{
    const uint b = get_global_id(0);
    const uint f = get_global_id(1);
    const uint y = get_global_id(2) / OUTPUT_SIZE_X;
    const uint x = get_global_id(2) % OUTPUT_SIZE_X;
    const uint output_idx = OUTPUT_GET_INDEX(b, f, y, x);

#if HAS_FUSED_OPS
    FUSED_OPS;
    output[output_idx] = FUSED_OPS_RESULT;
#endif
}
#endif
