// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

#if SOFTMAX_TOPK_ENABLE

KERNEL(softmax_topk)(
    const __global TYPE* input, // [input_batch, sort_in_num]
    __global uint* output_index, // [input_batch, TOP_K]
    __global TYPE* output // [input_batch, TOP_K]
) {
    // gws [batch, sort_in_num]
    const uint batch = (uint)get_global_id(0);
    const uint sort_index = (uint)get_global_id(1);
    const uint sort_cnt = (uint)get_global_size(1);

    input += batch * sort_cnt + sort_index;

    uint sort_position = 0;

    __local TYPE local_input[VALUE_NUM];
    __local TYPE local_output[TOP_K];
    __local uint local_index[TOP_K];

    TYPE in_value = as_half(intel_sub_group_block_read_us((const __global ushort*)(input)));
    local_input[sort_index] = in_value;
    barrier(CLK_LOCAL_MEM_FENCE);

    __attribute__((opencl_unroll_hint(8)))
    for(uint i = 0; i < sort_index; i++) {
        TYPE value = local_input[i];
        if(value >= in_value) {
            sort_position++;
        }
    }
    
    __attribute__((opencl_unroll_hint(8)))
    for(uint i = sort_index; i < sort_cnt; i++) {
        TYPE value = local_input[i];
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
        TYPE max_v = local_output[0];
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

#elif GATHER_ENABLE
KERNEL (gather_2d_ref)(
    const __global TYPE* src_tok,
    const __global TYPE* src_rweight,
    __global int * tok_index,
    __global int * top_index,
    __global TYPE* dst_tok,
    __global TYPE* dst_rweight) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];
    
    src_tok += tok_idx * HIDDEN_SIZE;
    dst_tok += k * HIDDEN_SIZE;

    #if TYPE_SIZE == 2
        ushort value = intel_sub_group_block_read_us((const __global ushort *)(src_tok + off));
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), value);
    #elif TYPE_SIZE == 4
        uint value = intel_sub_group_block_read((const __global uint *)(src_tok + off));
        intel_sub_group_block_write((__global uint *)(dst_tok + off), value);
    #else
        dst_tok[off] = src_tok[off];
    #endif

    if (off == 0) {
        int top_idx = top_index[k];
        dst_rweight[k] = src_rweight[top_idx];
    }
}

#elif SCATTER_ENABLE

KERNEL (index_add_)(const __global TYPE* src_tok,
    __global int * tok_index,
    __global TYPE* dst_tok) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];
    
    src_tok += k * HIDDEN_SIZE;
    dst_tok += tok_idx * HIDDEN_SIZE;

    #if TYPE_SIZE == 2
        half src_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(src_tok + off)));
        half dst_value = as_half(intel_sub_group_block_read_us((const __global ushort *)(dst_tok + off)));
        half value = dst_value + src_value;
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), as_ushort(value));
    #elif TYPE_SIZE == 4
        float src_value = as_float(intel_sub_group_block_read((const __global uint *)(src_tok + off)));
        float dst_value = as_float(intel_sub_group_block_read((const __global uint *)(dst_tok + off)));
        float value = dst_value + src_value;
        intel_sub_group_block_write_us((__global ushort *)(dst_tok + off), as_uint(value));
    #else
        dst_tok[off] += src_tok[off];
    #endif
}
#endif

