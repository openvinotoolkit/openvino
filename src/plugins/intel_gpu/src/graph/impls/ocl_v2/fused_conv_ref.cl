#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// FusedConv kernel: fuses Gather(beam_idx) + Concat + DepthwiseConv1D + SiLU + Slice
//
// Inputs:
//   INPUT0 (input):       [B, CONV_DIM, S]
//   INPUT1 (conv_weight): [CONV_DIM, KERNEL_SIZE]
//   INPUT2 (beam_idx):    [B]
//   INPUT3 (state_in):    [B, CONV_DIM, KERNEL_SIZE]
//
// Outputs:
//   OUTPUT  (output):     [B, CONV_DIM, S]
//   OUTPUT1 (state_out):  [B, CONV_DIM, KERNEL_SIZE]
//
// Dispatch: global = {batch, conv_dim, 1}, local = {1, WG_SIZE, 1}

KERNEL(fused_conv_ref)(OPTIONAL_SHAPE_INFO_ARG
    __global INPUT0_TYPE* input,
    __global INPUT1_TYPE* conv_weight,
    __global INPUT2_TYPE* beam_idx,
    __global INPUT3_TYPE* state_in,
    __global OUTPUT_TYPE* output,
    __global OUTPUT1_TYPE* state_out,
    int seq_len)
{
    const int b  = get_global_id(0);
    const int ch = get_global_id(1);

    if (ch >= CONV_DIM)
        return;

    // 1. Beam search reorder: read state from beam_idx[b] source batch
    const int src_b = convert_int(beam_idx[INPUT2_GET_INDEX(b, 0, 0, 0)]);

    // 2. Load state (KERNEL_SIZE values)
    float state[KERNEL_SIZE];
    for (int k = 0; k < KERNEL_SIZE; k++)
        state[k] = convert_float(state_in[INPUT3_GET_INDEX(src_b, ch, k, 0)]);

    // 3. Load conv weight
    float w[KERNEL_SIZE];
    for (int k = 0; k < KERNEL_SIZE; k++)
        w[k] = convert_float(conv_weight[INPUT1_GET_INDEX(ch, k, 0, 0)]);

    // 4. For each sequence position: depthwise conv + SiLU
    for (int s = 0; s < seq_len; s++) {
        const int input_idx = INPUT0_GET_INDEX(b, ch, s, 0);
        const int output_idx = OUTPUT_GET_INDEX(b, ch, s, 0);
        float x_new = convert_float(input[input_idx]);

        // Conv window = [state[1], state[2], ..., state[K-1], x_new]
        // This corresponds to concatenating state with input and applying valid conv
        float acc = 0.0f;
        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            acc += state[k + 1] * w[k];
        acc += x_new * w[KERNEL_SIZE - 1];

        // SiLU activation: x * sigmoid(x)
        float sig = native_recip(1.0f + native_exp(-acc));
        output[output_idx] = TO_OUTPUT_TYPE(acc * sig);

        // Shift state left, append new input
        for (int k = 0; k < KERNEL_SIZE - 1; k++)
            state[k] = state[k + 1];
        state[KERNEL_SIZE - 1] = x_new;
    }

    // 5. Write back updated state
    for (int k = 0; k < KERNEL_SIZE; k++)
        state_out[OUTPUT1_GET_INDEX(b, ch, k, 0)] = TO_OUTPUT1_TYPE(state[k]);
}
