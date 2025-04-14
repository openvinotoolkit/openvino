// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"


#if GATHER_ENABLE
__attribute__((intel_reqd_sub_group_size(16)))
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

__attribute__((intel_reqd_sub_group_size(16)))
KERNEL (index_add_)(const __global TYPE* src_tok,
    __global int * tok_index,
    __global TYPE* dst_tok) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];
    
    src_tok += k * HIDDEN_SIZE;
    dst_tok += tok_idx * HIDDEN_SIZE;

    dst_tok[off] += src_tok[off];
}
#endif

