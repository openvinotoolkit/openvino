// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q0 - SparsePA main K/V cache write.
//
// One subgroup copies the K and V row of one (token, kv head) into the paged cache. The input rows may be padded
// and offset (key_pitch / key_offset), so they are read lane-wise without any alignment assumption.
//
// JIT: QSA_KV_HEADS, QSA_K_HEAD_SIZE, QSA_V_HEAD_SIZE, QSA_PAGE_SIZE, SUBGROUP_SIZE
// Dispatch: gws = [SG, QSA_KV_HEADS, tokens_padded], lws = [SG, 1, 1]

#include "qsa_common.cl"

#if (QSA_K_HEAD_SIZE % SUBGROUP_SIZE) != 0 || (QSA_V_HEAD_SIZE % SUBGROUP_SIZE) != 0 || QSA_K_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE || QSA_V_HEAD_SIZE > QSA_MAX_NE * SUBGROUP_SIZE
#    error "head sizes must be whole subgroup rows of at most QSA_MAX_NE elements per lane"
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_kv_write)(const __global half* key,                   // [num_tokens, Hkv*Dh] token-major
                                  const __global half* value,                 // [num_tokens, Hkv*Dv] token-major
                                  const __global int* past_lens,              // [num_seqs]
                                  const __global int* block_indices,          // [used_blocks]
                                  const __global int* block_indices_begins,   // [num_seqs+1]
                                  const __global int* subsequence_begins,     // [num_seqs+1]
                                  __global half* key_cache,                   // [num_blocks, Hkv, page_size, Dh]
                                  __global half* value_cache,                 // [num_blocks, Hkv, page_size, Dv]
                                  const uint key_pitch,
                                  const uint key_offset,
                                  const uint value_pitch,
                                  const uint value_offset,
                                  const uint num_seqs) {
    const uint KV_HEADS = QSA_KV_HEADS;
    const uint K_HEAD = QSA_K_HEAD_SIZE;
    const uint V_HEAD = QSA_V_HEAD_SIZE;
    const uint PAGE = QSA_PAGE_SIZE;

    const uint head_idx = get_group_id(1);
    const uint token_idx = get_group_id(2);
    const uint lane = get_sub_group_local_id();

    if (token_idx >= (uint)subsequence_begins[num_seqs]) {
        return;
    }

    uint seq = 0;
    for (uint i = 0; i < num_seqs; i++) {
        if (token_idx >= (uint)subsequence_begins[i] && token_idx < (uint)subsequence_begins[i + 1]) {
            seq = i;
            break;
        }
    }

    const uint abs_pos = (uint)past_lens[seq] + token_idx - (uint)subsequence_begins[seq];
    const uint physical_page = (uint)block_indices[(uint)block_indices_begins[seq] + abs_pos / PAGE];
    const uint token_in_page = abs_pos % PAGE;

    const uint k_src = key_offset + token_idx * key_pitch + head_idx * K_HEAD;
    const uint v_src = value_offset + token_idx * value_pitch + head_idx * V_HEAD;
    const uint k_dst = FUNC_CALL(qsa_kv_cache_offset)(physical_page, head_idx, token_in_page, KV_HEADS, PAGE, K_HEAD);
    const uint v_dst = FUNC_CALL(qsa_kv_cache_offset)(physical_page, head_idx, token_in_page, KV_HEADS, PAGE, V_HEAD);

    // Every load is issued before the first store: the source and cache pointers may alias as far as the compiler
    // knows, so interleaving them would serialize one memory round trip per element.
    ushort kr[QSA_K_HEAD_SIZE / QSA_SG];
    ushort vr[QSA_V_HEAD_SIZE / QSA_SG];
    const __global ushort* ks = (const __global ushort*)key + k_src + lane;
    const __global ushort* vs = (const __global ushort*)value + v_src + lane;
    unroll_for (uint j = 0; j < K_HEAD / QSA_SG; j++) {
        kr[j] = ks[j * QSA_SG];
    }
    unroll_for (uint j = 0; j < V_HEAD / QSA_SG; j++) {
        vr[j] = vs[j * QSA_SG];
    }
    FUNC_CALL(qsa_wr)((__global ushort*)(key_cache + k_dst), kr, K_HEAD / QSA_SG);
    FUNC_CALL(qsa_wr)((__global ushort*)(value_cache + v_dst), vr, V_HEAD / QSA_SG);
}
