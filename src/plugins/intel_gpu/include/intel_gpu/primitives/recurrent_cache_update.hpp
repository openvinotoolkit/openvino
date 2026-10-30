// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include "primitive.hpp"
namespace cldnn {
// Dedicated Qwen llama.cpp primitive. Positive history selects convolution; zero selects GDN.
struct recurrent_cache_update : primitive_base<recurrent_cache_update> {
    CLDNN_DECLARE_PRIMITIVE(recurrent_cache_update)
    recurrent_cache_update() : primitive_base("", {}) {}
    recurrent_cache_update(const primitive_id& id,
                           const std::vector<input_info>& inputs,
                           int64_t history = 0,
                           bool norm = false,
                           float qe = 1e-6f,
                           float ke = 1e-6f,
                           int64_t count = 1)
        : primitive_base(id, inputs),
          history(history),
          fuse_qk_l2norm(norm),
          q_l2_norm_eps(qe),
          k_l2_norm_eps(ke),
          snapshot_count(count) {
        num_outputs = 2;
    }
    size_t hash() const override {
        size_t s = primitive::hash();
        s = hash_combine(s, history);
        s = hash_combine(s, fuse_qk_l2norm);
        s = hash_combine(s, q_l2_norm_eps);
        s = hash_combine(s, k_l2_norm_eps);
        return hash_combine(s, snapshot_count);
    }
    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto p = downcast<const recurrent_cache_update>(rhs);
        return history == p.history && fuse_qk_l2norm == p.fuse_qk_l2norm && q_l2_norm_eps == p.q_l2_norm_eps && k_l2_norm_eps == p.k_l2_norm_eps &&
               snapshot_count == p.snapshot_count;
    }
    void save(BinaryOutputBuffer& b) const override {
        primitive_base<recurrent_cache_update>::save(b);
        b << history;
        b << fuse_qk_l2norm;
        b << q_l2_norm_eps;
        b << k_l2_norm_eps;
        b << snapshot_count;
    }
    void load(BinaryInputBuffer& b) override {
        primitive_base<recurrent_cache_update>::load(b);
        b >> history;
        b >> fuse_qk_l2norm;
        b >> q_l2_norm_eps;
        b >> k_l2_norm_eps;
        b >> snapshot_count;
    }
    int64_t history = 0;
    bool fuse_qk_l2norm = false;
    float q_l2_norm_eps = 1e-6f, k_l2_norm_eps = 1e-6f;
    int64_t snapshot_count = 1;
};
}  // namespace cldnn
