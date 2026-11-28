// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// out[r, d, e] = y[r, e] * post[r, d] + sum_s comb[r, s, d] * res[r, s, e]   (see ov::op::internal::HyperConnectionPost)
KERNEL(hyper_connection_post_ref)
(const __global INPUT0_TYPE* y,
 const __global INPUT1_TYPE* res,
 const __global INPUT2_TYPE* post,
 const __global INPUT3_TYPE* comb,
 __global OUTPUT_TYPE* out) {
    const size_t e = get_global_id(0);  // size_t: rows * HC_N * HC_E can exceed INT_MAX on long prefills
    const size_t r = get_global_id(1);
    float rv[HC_N];
    for (int s = 0; s < HC_N; s++)
        rv[s] = convert_float(res[(r * HC_N + s) * HC_E + e]);
    const float yv = convert_float(y[r * HC_E + e]);
    for (int d = 0; d < HC_N; d++) {
        float acc = yv * convert_float(post[r * HC_N + d]);
        for (int s = 0; s < HC_N; s++)
            acc = fma(convert_float(comb[(r * HC_N + s) * HC_N + d]), rv[s], acc);
        out[(r * HC_N + d) * HC_E + e] = TO_OUTPUT_TYPE(acc);
    }
}
