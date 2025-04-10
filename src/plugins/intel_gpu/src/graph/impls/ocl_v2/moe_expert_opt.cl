// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"


#if GATHER_ENABLE

KERNEL (gather_2d_ref)(
    const __global half* src_tok,
    const __global half* src_rweight,
    __global int * tok_index,
    __global int * top_index,
    __global half* dst_tok,
    __global half* dst_rweight) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];
    
    src_tok += tok_idx * HIDDEN_SIZE;
    dst_tok += k * HIDDEN_SIZE;

    dst_tok[off] = src_tok[off];

    if (off == 0) {
        int top_idx = top_index[k];
        dst_rweight[k] = src_rweight[top_idx];
    }
}

#elif SCATTER_ENABLE

KERNEL (index_add_)(const __global half* src_tok,
    __global int * tok_index,
    __global half* dst_tok) {

    int k = get_global_id(0);
    int off = get_global_id(1);
    int tok_idx = tok_index[k];
    
    src_tok += k * HIDDEN_SIZE;
    dst_tok += tok_idx * HIDDEN_SIZE;

    dst_tok[off] += src_tok[off];
}
#endif

