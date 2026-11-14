// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// QSAIndexer (non-paged) block-key cache carry.
//
// The output cache holds max(key_length) / r blocks, while the input cache holds the blocks of the earlier calls and
// may have any larger block count. Per row, the blocks before past_len / r are copied from the input, the blocks after
// key_length / r are zero-filled, and the blocks in between are left to qsa_indexer_prepare_dense.
//
// JIT: QSA_IDX_HEAD_DIM, QSA_COMPRESS_RATIO
// Dispatch: gws = [Di, out_blocks, B]

#include "include/batch_headers/common.cl"

KERNEL(qsa_indexer_carry_cache_dense)(const __global int* key_length,  // [B]
                                      const __global half* cache_in,   // [B, 1, in_blocks, Di]
                                      __global half* cache_out,        // [B, 1, out_blocks, Di]
                                      const uint seq_len,
                                      const uint in_blocks,
                                      const uint out_blocks) {
    const uint DI = QSA_IDX_HEAD_DIM;
    const uint R = QSA_COMPRESS_RATIO;

    const uint d = get_global_id(0);
    const uint block = get_global_id(1);
    const uint batch = get_global_id(2);
    if (d >= DI || block >= out_blocks) {
        return;
    }

    const uint total_len = (uint)key_length[batch];
    const uint n_new = total_len < seq_len ? total_len : seq_len;
    const uint past_blocks = (total_len - n_new) / R;
    if (block >= past_blocks && block < total_len / R) {
        return;
    }
    half value = 0;
    if (block < past_blocks && block < in_blocks) {
        value = cache_in[(batch * in_blocks + block) * DI + d];
    }
    cache_out[(batch * out_blocks + block) * DI + d] = value;
}
