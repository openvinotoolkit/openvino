// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// mHC pre side (ov::op::internal::HyperConnectionPre), two stages.
// Stage A (HC_PRE_MIXES): HC_KSPLIT work-groups per (row, logit o), each over one K chunk: partial <fn[o], x> and
//   partial sum(x^2), x = the row's flattened streams [HC_N * HC_E]. The RMS scale factors out of the dot product:
//   mixes[o] = rsqrt(sum(x^2) / K + rms_eps) * <fn[o], x>, finished in stage B. (2+n)*n*HC_KSPLIT work-groups per row.
// Stage B (HC_PRE_MIXSUM): one work-group per row: subgroup 0 computes pre/post/comb exactly as hyper_connection_mix
//   (n == 4: one lane per comb entry), then every lane computes y[c] = sum_i pre[i] * x[i][c].

#define HC_K (HC_N * HC_E)

#ifdef HC_PRE_MIXES
// work-group (o * HC_KSPLIT + part, row): partial <fn[o], x> and partial sum of squares over K chunk `part`
KERNEL(hyper_connection_pre_mixes)
(const __global INPUT0_TYPE* streams,
 const __global INPUT1_TYPE* fn,
 __global float* part_buf,
 int rows) {
    const int g = get_group_id(1);
    const int o = g / HC_KSPLIT;
    const int part = g % HC_KSPLIT;
    const int r = get_group_id(2);
    const int lid = get_local_id(0);
    if (r >= rows)
        return;
    const int chunk = HC_K / HC_KSPLIT;
    const __global INPUT0_TYPE* x = streams + (size_t)r * HC_K + part * chunk;
    const __global INPUT1_TYPE* w = fn + (size_t)o * HC_K + part * chunk;
    float ss = 0.0f, acc = 0.0f;
    for (int j = lid; j < chunk; j += HC_LWS) {
        const float v = convert_float(x[j]);
        ss = fma(v, v, ss);
        acc = fma(convert_float(w[j]), v, acc);
    }
    ss = work_group_reduce_add(ss);
    acc = work_group_reduce_add(acc);
    if (lid == 0) {
        __global float* b = part_buf + ((size_t)r * HC_MIX + o) * (2 * HC_KSPLIT);
        b[part] = acc;
        b[HC_KSPLIT + part] = ss;
    }
}
#endif

#ifdef HC_PRE_MIXSUM
REQD_SUB_GROUP_SIZE(16)
KERNEL(hyper_connection_pre_mixsum)
(const __global INPUT0_TYPE* streams,
 const __global INPUT2_TYPE* scale,
 const __global INPUT3_TYPE* base,
 const __global float* mixes,
 __global OUTPUT_TYPE* y,
 __global OUTPUT1_TYPE* post,
 __global OUTPUT2_TYPE* comb,
 int rows) {
    __local float pre_s[HC_N];
    const int r = get_group_id(0);
    const int lid = get_local_id(0);
    if (r >= rows)
        return;
    __local float m[HC_MIX];
    if (lid < HC_MIX) {
        const __global float* b = mixes + ((size_t)r * HC_MIX + lid) * (2 * HC_KSPLIT);
        float acc = 0.0f, ss = 0.0f;
        for (int p = 0; p < HC_KSPLIT; p++) {
            acc += b[p];
            ss += b[HC_KSPLIT + p];
        }
        m[lid] = acc * rsqrt(ss / (float)HC_K + HC_RMS_EPS);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const float eps = HC_EPS;
    if (get_sub_group_id() == 0) {
        const int l = get_sub_group_local_id();
        if (l < HC_N) {
            const float a = m[l] * convert_float(scale[0]) + convert_float(base[l]);
            pre_s[l] = 1.0f / (1.0f + exp(-a)) + eps;
        } else if (l < 2 * HC_N) {
            const int i = l - HC_N;
            const float b = m[HC_N + i] * convert_float(scale[1]) + convert_float(base[HC_N + i]);
            post[r * HC_N + i] = TO_OUTPUT1_TYPE(2.0f / (1.0f + exp(-b)));
        }
        float c = m[2 * HC_N + l] * convert_float(scale[2]) + convert_float(base[2 * HC_N + l]);
        float mx = fmax(c, intel_sub_group_shuffle(c, l ^ 1));
        mx = fmax(mx, intel_sub_group_shuffle(mx, l ^ 2));
        c = exp(c - mx);
        float s = c + intel_sub_group_shuffle(c, l ^ 1);
        s += intel_sub_group_shuffle(s, l ^ 2);
        c = c / s + eps;
        for (int it = 0; it < HC_ITERS; it++) {
            if (it > 0) {
                float rs = c + intel_sub_group_shuffle(c, l ^ 1);
                rs += intel_sub_group_shuffle(rs, l ^ 2);
                c = c / (rs + eps);
            }
            float cs = c + intel_sub_group_shuffle(c, l ^ 4);
            cs += intel_sub_group_shuffle(cs, l ^ 8);
            c = c / (cs + eps);
        }
        comb[r * HC_N * HC_N + l] = TO_OUTPUT2_TYPE(c);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    const __global INPUT0_TYPE* x = streams + (size_t)r * HC_K;
    const float p0 = pre_s[0], p1 = pre_s[1], p2 = pre_s[2], p3 = pre_s[3];
    for (int c = lid; c < HC_E; c += HC_LWS) {
        const float acc = p0 * convert_float(x[c]) + p1 * convert_float(x[HC_E + c]) + p2 * convert_float(x[2 * HC_E + c]) +
                          p3 * convert_float(x[3 * HC_E + c]);
        y[(size_t)r * HC_E + c] = TO_OUTPUT_TYPE(acc);
    }
}
#endif
