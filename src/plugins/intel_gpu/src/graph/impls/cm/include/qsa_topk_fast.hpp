// Copyright (c) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#ifndef QSA_TOPK_FAST_HPP
#define QSA_TOPK_FAST_HPP
#include "qsa_score_common.hpp"

CM_INLINE vector<uint, 64> fast_keys(const float* scores ATTR, uint i, uint n) {
    vector<uint, 64> lane;
    cmtl::cm_vector_assign(lane.select_all(), 0, 1);
    vector<float, 64> f;
    if (i + 64 <= n) f = cm_ptr_load<float, 64>(scores, i * 4);
    else f = cm_ptr_load<float>(scores, (lane + i) * 4, lane + i < n);
    vector<uint, 64> bits = f.format<uint>();
    vector<uint, 64> mask = 0x80000000u;
    mask.merge(0xFFFFFFFFu, (bits & 0x80000000u) != 0);
    return bits ^ mask;
}

CM_INLINE vector<uint, 64> fast_scan(vector<uint, 64> x) {
    vector<uint, 64> y = x;
    y.select<63, 1>(1) = x.select<63, 1>(1) + x.select<63, 1>(0); x = y;
    y.select<62, 1>(2) = x.select<62, 1>(2) + x.select<62, 1>(0); x = y;
    y.select<60, 1>(4) = x.select<60, 1>(4) + x.select<60, 1>(0); x = y;
    y.select<56, 1>(8) = x.select<56, 1>(8) + x.select<56, 1>(0); x = y;
    y.select<48, 1>(16) = x.select<48, 1>(16) + x.select<48, 1>(0); x = y;
    y.select<32, 1>(32) = x.select<32, 1>(32) + x.select<32, 1>(0);
    return y;
}
#endif