// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Speculative-decoding tree mask shared by the CM paged-attention kernels.
//
// qq_bias layout: u8 row-major [spec_num, spec_num], indexed as
// qq_bias[query_spec * spec_num + key_spec]; a zero entry means the pair is masked.
// The mask applies only to the "new" K range (key_spec >= 0) and to queries that
// fall inside the spec window (query_spec in [0, spec_num)).
//
// This header is intentionally free of tile-geometry macros (REG_N, kv_step, ...)
// and jit constants so kernels with their own geometry can include it directly.

// Transposed layout: St[k_row][q_col], k_row in [0, N), q_col in [0, M).
template <int N, int M>
inline void apply_qq_bias_tree_mask(matrix_ref<float, N, M> St,
                                    svmptr_t qq_bias_base,
                                    int qq_bias_spec_num,
                                    int kv_pos,
                                    int q_start,
                                    int past_lens) {
    if (qq_bias_spec_num <= 0) return;
    const uchar* qq_bias_ptr = reinterpret_cast<const uchar*>(qq_bias_base);
    #pragma unroll
    for (int k_row = 0; k_row < N; ++k_row) {
        const int key_local = kv_pos + k_row;
        const int key_spec = key_local - past_lens;
        if (key_spec < 0 || key_spec >= qq_bias_spec_num) continue;
        #pragma unroll
        for (int q_col = 0; q_col < M; ++q_col) {
            const int query_spec = q_start + q_col;
            if (query_spec < 0 || query_spec >= qq_bias_spec_num) continue;
            const int qq_off = query_spec * qq_bias_spec_num + key_spec;
            if (qq_bias_ptr[qq_off] == 0) {
                St[k_row][q_col] = -3.4e38f;
            }
        }
    }
}

// Query-major counterpart, for kernels whose S tile is laid out S[q_row][k_col].
// Builds the merge mask for one query row over K contiguous keys starting at
// key_local_start; a lane is set when the (query_spec, key_spec) pair is masked out.
// Lanes whose key falls outside the spec window stay clear, so the caller can merge
// the result unconditionally. The mask depends only on query_spec, so callers that
// replicate a query across GQA heads should build it once and reuse it per head.
template <int K>
inline vector<unsigned short, K> build_qq_bias_tree_mask_row(const uchar* qq_bias_ptr,
                                                             int qq_bias_spec_num,
                                                             int query_spec,
                                                             int key_local_start,
                                                             int past_lens) {
    vector<unsigned short, K> m = 0;
    if (qq_bias_spec_num <= 0) return m;
    if (query_spec < 0 || query_spec >= qq_bias_spec_num) return m;
    const int row_off = query_spec * qq_bias_spec_num;
    #pragma unroll
    for (int c = 0; c < K; ++c) {
        const int key_spec = key_local_start + c - past_lens;
        if (key_spec < 0 || key_spec >= qq_bias_spec_num) continue;
        if (qq_bias_ptr[row_off + key_spec] == 0) m[c] = 1;
    }
    return m;
}
