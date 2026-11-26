// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// mHC coefficients per token row (see ov::op::internal::HyperConnectionMix), c[src * HC_N + dst].
// The Sinkhorn loop is latency-bound on one work-item (~31 us on Arc B390 for n = 4, 20 iterations), so for
// HC_N == 4 one 16-lane subgroup handles a row: lane l holds c[l] (src = l / 4, dst = l % 4); sums over dst
// shuffle across lanes xor 1, 2 and sums over src across lanes xor 4, 8.
#if HC_N == 4
REQD_SUB_GROUP_SIZE(16)
KERNEL(hyper_connection_mix_ref)
(const __global INPUT0_TYPE* mixes,
 const __global INPUT1_TYPE* scale,
 const __global INPUT2_TYPE* base,
 __global OUTPUT_TYPE* pre,
 __global OUTPUT1_TYPE* post,
 __global OUTPUT2_TYPE* comb,
 int rows) {
    const int r = get_group_id(0);
    const int l = get_sub_group_local_id();
    if (r >= rows)
        return;
    const __global INPUT0_TYPE* m = mixes + r * HC_MIX;
    const float eps = HC_EPS;

    if (l < HC_N) {
        const float a = convert_float(m[l]) * convert_float(scale[0]) + convert_float(base[l]);
        pre[r * HC_N + l] = TO_OUTPUT_TYPE(1.0f / (1.0f + exp(-a)) + eps);
    } else if (l < 2 * HC_N) {
        const int i = l - HC_N;
        const float b = convert_float(m[HC_N + i]) * convert_float(scale[1]) + convert_float(base[HC_N + i]);
        post[r * HC_N + i] = TO_OUTPUT1_TYPE(2.0f / (1.0f + exp(-b)));
    }

    float c = convert_float(m[2 * HC_N + l]) * convert_float(scale[2]) + convert_float(base[2 * HC_N + l]);
    // softmax over dst (the 4 lanes of one src), then + eps
    float mx = fmax(c, intel_sub_group_shuffle(c, l ^ 1));
    mx = fmax(mx, intel_sub_group_shuffle(mx, l ^ 2));
    c = exp(c - mx);
    float s = c + intel_sub_group_shuffle(c, l ^ 1);
    s += intel_sub_group_shuffle(s, l ^ 2);
    c = c / s + eps;

    for (int it = 0; it < HC_ITERS; it++) {
        if (it > 0) {
            // rows: divide by the sum over dst + eps
            float rs = c + intel_sub_group_shuffle(c, l ^ 1);
            rs += intel_sub_group_shuffle(rs, l ^ 2);
            c = c / (rs + eps);
        }
        // cols: divide by the sum over src + eps
        float cs = c + intel_sub_group_shuffle(c, l ^ 4);
        cs += intel_sub_group_shuffle(cs, l ^ 8);
        c = c / (cs + eps);
    }
    comb[r * HC_N * HC_N + l] = TO_OUTPUT2_TYPE(c);
}
#else
KERNEL(hyper_connection_mix_ref)
(const __global INPUT0_TYPE* mixes,
 const __global INPUT1_TYPE* scale,
 const __global INPUT2_TYPE* base,
 __global OUTPUT_TYPE* pre,
 __global OUTPUT1_TYPE* post,
 __global OUTPUT2_TYPE* comb,
 int rows) {
    const int r = get_global_id(0);
    if (r >= rows)
        return;
    const __global INPUT0_TYPE* m = mixes + r * HC_MIX;
    const float s0 = convert_float(scale[0]);
    const float s1 = convert_float(scale[1]);
    const float s2 = convert_float(scale[2]);
    const float eps = HC_EPS;
    for (int i = 0; i < HC_N; i++) {
        const float a = convert_float(m[i]) * s0 + convert_float(base[i]);
        const float b = convert_float(m[HC_N + i]) * s1 + convert_float(base[HC_N + i]);
        pre[r * HC_N + i] = TO_OUTPUT_TYPE(1.0f / (1.0f + exp(-a)) + eps);
        post[r * HC_N + i] = TO_OUTPUT1_TYPE(2.0f / (1.0f + exp(-b)));
    }
    float c[HC_N * HC_N];
    for (int k = 0; k < HC_N * HC_N; k++)
        c[k] = convert_float(m[2 * HC_N + k]) * s2 + convert_float(base[2 * HC_N + k]);
    for (int s = 0; s < HC_N; s++) {
        float mx = c[s * HC_N];
        for (int d = 1; d < HC_N; d++)
            mx = fmax(mx, c[s * HC_N + d]);
        float sum = 0.0f;
        for (int d = 0; d < HC_N; d++) {
            c[s * HC_N + d] = exp(c[s * HC_N + d] - mx);
            sum += c[s * HC_N + d];
        }
        for (int d = 0; d < HC_N; d++)
            c[s * HC_N + d] = c[s * HC_N + d] / sum + eps;
    }
    for (int it = 0; it < HC_ITERS; it++) {
        if (it > 0) {
            for (int s = 0; s < HC_N; s++) {
                float sum = eps;
                for (int d = 0; d < HC_N; d++)
                    sum += c[s * HC_N + d];
                for (int d = 0; d < HC_N; d++)
                    c[s * HC_N + d] /= sum;
            }
        }
        for (int d = 0; d < HC_N; d++) {
            float sum = eps;
            for (int s = 0; s < HC_N; s++)
                sum += c[s * HC_N + d];
            for (int s = 0; s < HC_N; s++)
                c[s * HC_N + d] /= sum;
        }
    }
    for (int k = 0; k < HC_N * HC_N; k++)
        comb[r * HC_N * HC_N + k] = TO_OUTPUT2_TYPE(c[k]);
}
#endif
