// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if SOFTMAX_TOPK_ENABLE

KERNEL(softmax_topk)(
    const __global MOE_DTYPE* input, // [input_batch, sort_in_num]
    __global uint* output_index, // [input_batch, TOP_K]
    __global MOE_DTYPE* output // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;

    uint sort_position = 0;

    __local MOE_DTYPE local_input[VALUE_NUM];
    __local MOE_DTYPE local_output[TOP_K];
    __local uint local_index[TOP_K];

#if MOE_DTYPE_SIZE == 2
    MOE_DTYPE in_value = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
#elif MOE_DTYPE_SIZE == 4
    MOE_DTYPE in_value = as_float(intel_sub_group_block_read((const __global uint*)(input)));
#else
#    error "softmax_topk: unsupported MOE_DTYPE_SIZE"
#endif
    local_input[sort_index] = in_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        MOE_DTYPE value = local_input[i];
        if(value >= in_value) {
            sort_position++;
        }
    }

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        MOE_DTYPE value = local_input[i];
        if(value > in_value) {
            sort_position++;
        }
    }
    if (sort_position < TOP_K) {
        local_output[sort_position] = in_value;
        local_index[sort_position] = sort_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(sort_position == 0) {
        float softmax_total = 1.0;
        MOE_DTYPE max_v = local_output[0];
        local_output[0] = 1;
        for(uint i = 1; i < TOP_K; i++) {
            local_output[i] = native_exp(local_output[i] - max_v);
            softmax_total += local_output[i];
        }
        output_index += batch * TOP_K;
        output += batch * TOP_K;

        for(uint i = 0; i < TOP_K; i++) {
            output[i] = local_output[i]/softmax_total;
            output_index[i] = local_index[i];
        }
    }
}

#elif SIGMOID_BIAS_TOPK_ENABLE

KERNEL(sigmoid_bias_topk)(
    const __global MOE_DTYPE* input,    // routing logits [input_batch, num_experts]
    const __global MOE_DTYPE* bias,     // routing bias [1, num_experts] or [num_experts]
    const __global MOE_DTYPE* eps_ptr,  // routing epsilon scalar [1]
    __global uint* output_index,        // [input_batch, TOP_K]
    __global MOE_DTYPE* output          // [input_batch, TOP_K]
) {
    // gws [batch, num_experts]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);  // num_experts

    input += batch * sort_cnt + sort_index;

    __local MOE_DTYPE local_sigmoid[VALUE_NUM];     // raw sigmoid values
    __local MOE_DTYPE local_selection[VALUE_NUM];   // sigmoid + bias (for sorting)
    __local MOE_DTYPE local_output[TOP_K];
    __local uint local_index[TOP_K];

    // Compute sigmoid
#if MOE_DTYPE_SIZE == 2
    MOE_DTYPE in_value = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
#elif MOE_DTYPE_SIZE == 4
    MOE_DTYPE in_value = as_float(intel_sub_group_block_read((const __global uint*)(input)));
#else
#    error "sigmoid_bias_topk: unsupported MOE_DTYPE_SIZE"
#endif
    MOE_DTYPE sigmoid_val = (MOE_DTYPE)(1.0f / (1.0f + native_exp(-(float)in_value)));

    // Add bias for selection (determines which experts are chosen)
    MOE_DTYPE bias_val = bias[sort_index];
    MOE_DTYPE selection_val = sigmoid_val + bias_val;

    local_sigmoid[sort_index] = sigmoid_val;
    local_selection[sort_index] = selection_val;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sort by selection_val (sigmoid + bias) to find top-K
    uint sort_position = 0;
    uint actual_topk = (TOP_K < sort_cnt) ? TOP_K : sort_cnt;

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        MOE_DTYPE value = local_selection[i];
        if(value >= selection_val) {
            sort_position++;
        }
    }

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        MOE_DTYPE value = local_selection[i];
        if(value > selection_val) {
            sort_position++;
        }
    }

    // Store raw sigmoid values (NOT sigmoid+bias) for the top-K experts
    if (sort_position < actual_topk) {
        local_output[sort_position] = local_sigmoid[sort_index];
        local_index[sort_position] = sort_index;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Normalize: weights / (sum + eps)
    if(sort_position == 0) {
        float sum_weights = 0.0f;
        for(uint i = 0; i < actual_topk; i++) {
            sum_weights += (float)local_output[i];
        }
        sum_weights += (float)eps_ptr[0];  // epsilon to avoid division by zero

        output_index += batch * TOP_K;
        output += batch * TOP_K;

        for(uint i = 0; i < actual_topk; i++) {
            output[i] = (MOE_DTYPE)((float)local_output[i] / sum_weights);
            output_index[i] = local_index[i];
        }
        // Zero out remaining positions if TOP_K > actual_topk
        for(uint i = actual_topk; i < TOP_K; i++) {
            output[i] = (MOE_DTYPE)0.0f;
            output_index[i] = 0;
        }
    }
}

#endif
