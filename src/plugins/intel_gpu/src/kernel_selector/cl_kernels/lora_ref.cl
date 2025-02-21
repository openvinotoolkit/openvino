// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if LORA_COUNT == 1
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
        uint state_a_idx = local_id * K + ki;

        acc = mad(TO_ACCUMULATOR_TYPE(lora_input[lora_input_idx]),
                  TO_ACCUMULATOR_TYPE(state_a[state_a_idx]),
                  acc);
    }

    acc *= TO_ACCUMULATOR_TYPE(state_alpha[local_id]);

    __local ACCUMULATOR_TYPE tmp_buf[MAX_LORA_RANK];
    tmp_buf[local_id] = acc;

    barrier(CLK_LOCAL_MEM_FENCE);

    ACCUMULATOR_TYPE final_acc = ACCUMULATOR_VAL_ZERO;
    uint processing_size = INPUT0_SIZE_Y * INPUT0_SIZE_X / LORA_RANK;
    uint new_yx = yx + processing_size * local_id;

    for (uint ki = 0; ki < LORA_RANK; ++ki) {
        uint state_b_idx = new_yx * LORA_RANK + ki;

        final_acc = mad(tmp_buf[ki], TO_ACCUMULATOR_TYPE(state_b[state_b_idx]), final_acc);
    }

    uint output_idx = bf * INPUT0_SIZE_Y + new_yx;
    output[output_idx] = TO_OUTPUT_TYPE(final_acc) + main_input[output_idx];

    for (uint leftover = processing_size * LORA_RANK + local_id; leftover < INPUT0_SIZE_Y * INPUT0_SIZE_X; leftover += LORA_RANK) {
        final_acc = ACCUMULATOR_VAL_ZERO;
        for (uint ki = 0; ki < LORA_RANK; ++ki) {
            uint state_b_idx = leftover * LORA_RANK + ki;
            final_acc = mad(tmp_buf[ki], TO_ACCUMULATOR_TYPE(state_b[state_b_idx]), final_acc);
        }
        output_idx = bf * INPUT0_SIZE_Y + leftover;
        output[output_idx] = TO_OUTPUT_TYPE(final_acc) + main_input[output_idx];
    }
}

#elif LORA_COUNT == 2

KERNEL(lora_ref)(OPTIONAL_SHAPE_INFO_ARG
                 const __global OUTPUT_TYPE* main_input,
                 const __global INPUT1_TYPE* lora_input,
                 const __global STATE_TYPE* state_a_0,
                 const __global STATE_TYPE* state_alpha_0,
                 const __global STATE_TYPE* state_b_0,
                 const __global STATE_TYPE* state_a_1,
                 const __global STATE_TYPE* state_alpha_1,
                 const __global STATE_TYPE* state_b_1,
                       __global OUTPUT_TYPE* output)
{
    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        printf("LORA_COUNT = 2\n");
    }
}

#elif LORA_COUNT == 3

KERNEL(lora_ref)(OPTIONAL_SHAPE_INFO_ARG
                 const __global OUTPUT_TYPE* main_input,
                 const __global INPUT1_TYPE* lora_input,
                 const __global STATE_TYPE* state_a_0,
                 const __global STATE_TYPE* state_alpha_0,
                 const __global STATE_TYPE* state_b_0,
                 const __global STATE_TYPE* state_a_1,
                 const __global STATE_TYPE* state_alpha_1,
                 const __global STATE_TYPE* state_b_1,
                 const __global STATE_TYPE* state_a_2,
                 const __global STATE_TYPE* state_alpha_2,
                 const __global STATE_TYPE* state_b_2,
                       __global OUTPUT_TYPE* output)
{
    if (get_global_id(0) == 0 && get_global_id(1) == 0 && get_global_id(2) == 0) {
        printf("LORA_COUNT = 3\n");
    }
}

#endif
