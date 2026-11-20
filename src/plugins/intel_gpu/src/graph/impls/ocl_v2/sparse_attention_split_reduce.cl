// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Q3 decode, split-selected-KV variant, pass 2 - merge.
//
// Merges the (max, sum, weighted-V) state of every partition with the online-softmax combine rule. The maxima are in
// natural-log units. An empty partition (sum <= 0) is skipped BEFORE exponentiating so its possibly garbage max can
// not contaminate the running max.
//
// JIT: QSA_HEADS, QSA_V_HEAD_SIZE, QSA_SPLIT_PARTITIONS, QSA_REDUCE_COLS, QSA_ADDRESS_MODE, SUBGROUP_SIZE
// Dispatch: gws = [QSA_HEADS * SG, num_rows, QSA_V_HEAD_SIZE / QSA_REDUCE_COLS], lws = [SG, 1, 1]

#include "qsa_common.cl"

#ifndef QSA_ADDRESS_MODE
#    define QSA_ADDRESS_MODE 0
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(sparse_attention_split_reduce)(const __global float* partial_acc,  // [rows, Hq, P, Dv]
                                      const __global float* partial_ml,   // [rows, Hq, P, 2]
                                      __global half* output,              // paged: [rows, Hq*Dv]; dense: [B, Hq, L, Dv]
#if QSA_ADDRESS_MODE == 0
                                      const uint num_rows
#else
                                      const uint seq_len,
                                      const uint num_rows
#endif
) {
    const uint HQ = QSA_HEADS;
    const uint DV = QSA_V_HEAD_SIZE;
    const uint P = QSA_SPLIT_PARTITIONS;
    const uint COLS = QSA_REDUCE_COLS;
    const uint lane = get_sub_group_local_id();

    const uint head = get_group_id(0);
    const uint row = get_group_id(1);
    const uint col0 = get_group_id(2) * COLS;
    if (head >= HQ || row >= num_rows || col0 >= DV) {
        return;
    }

    const uint row0 = (row * HQ + head) * P;
    const uint PG = (QSA_SPLIT_PARTITIONS + QSA_SG - 1) / QSA_SG;

    // Lane i of group g holds partition g * SG + i; an empty partition contributes neither max nor weight.
    float m[(QSA_SPLIT_PARTITIONS + QSA_SG - 1) / QSA_SG];
    float l[(QSA_SPLIT_PARTITIONS + QSA_SG - 1) / QSA_SG];
    float m_max = -1e30f;
    unroll_for (uint g = 0; g < PG; g++) {
        const uint p = g * QSA_SG + lane;
        const float2 ml = p < P ? vload2(row0 + p, partial_ml) : (float2)(-1e30f, 0.0f);
        l[g] = ml.s1;
        m[g] = ml.s1 > 0.0f ? ml.s0 : -1e30f;
        m_max = fmax(m_max, m[g]);
    }
    m_max = sub_group_reduce_max(m_max);

    float w[(QSA_SPLIT_PARTITIONS + QSA_SG - 1) / QSA_SG];
    float sum = 0.0f;
    unroll_for (uint g = 0; g < PG; g++) {
        w[g] = l[g] > 0.0f ? native_exp2((m[g] - m_max) * QSA_LOG2E) : 0.0f;
        sum += w[g] * l[g];
    }
    sum = sub_group_reduce_add(sum);

    float acc[QSA_REDUCE_COLS / QSA_SG];
    unroll_for (uint j = 0; j < COLS / QSA_SG; j++) {
        acc[j] = 0.0f;
    }
    // Every producer writes all P rows (zeros when empty), so unconditional loads are safe.
    unroll_for (uint p = 0; p < P; p++) {
        const float wp = sub_group_broadcast(w[p / QSA_SG], p % QSA_SG);
        const __global uint* src = (const __global uint*)(partial_acc + (row0 + p) * DV + col0);
#if QSA_REDUCE_COLS == 2 * SUBGROUP_SIZE
        const float2 a = as_float2(intel_sub_group_block_read2(src));
        acc[0] += a.s0 * wp;
        acc[1] += a.s1 * wp;
#else
        unroll_for (uint j = 0; j < COLS / QSA_SG; j++) {
            acc[j] += as_float(intel_sub_group_block_read(src + j * QSA_SG)) * wp;
        }
#endif
    }

    const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
    float o[QSA_REDUCE_COLS / QSA_SG];
    unroll_for (uint j = 0; j < COLS / QSA_SG; j++) {
        o[j] = acc[j] * inv_sum;
    }
#if QSA_ADDRESS_MODE == 0
    const uint out_row = row * HQ + head;
#else
    const uint batch = row / seq_len;
    const uint out_row = (batch * HQ + head) * seq_len + (row - batch * seq_len);
#endif
    FUNC_CALL(qsa_wr_f32)(output + out_row * DV + col0, o, COLS / QSA_SG);
}
