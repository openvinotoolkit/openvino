// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "primitive.hpp"

namespace cldnn {
struct recurrent_cache_replay : primitive_base<recurrent_cache_replay> {
    CLDNN_DECLARE_PRIMITIVE(recurrent_cache_replay)
    recurrent_cache_replay() : primitive_base("", {}) {}
    recurrent_cache_replay(const primitive_id& id, const std::vector<input_info>& inputs, int64_t history = 0,
                           bool norm = false, float qe = 1e-6f, float ke = 1e-6f, bool grouped = false)
        : primitive_base(id, inputs), history(history), fuse_qk_l2norm(norm), q_l2_norm_eps(qe), k_l2_norm_eps(ke),
          grouped_heads(grouped) {
        num_outputs = history > 0 ? 2 : 5;
    }
    size_t hash() const override {
        auto value = primitive::hash();
        value = hash_combine(value, history);
        value = hash_combine(value, fuse_qk_l2norm);
        value = hash_combine(value, q_l2_norm_eps);
        value = hash_combine(value, k_l2_norm_eps);
        return hash_combine(value, grouped_heads);
    }
    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        const auto& other = downcast<const recurrent_cache_replay>(rhs);
        return history == other.history && fuse_qk_l2norm == other.fuse_qk_l2norm &&
               q_l2_norm_eps == other.q_l2_norm_eps && k_l2_norm_eps == other.k_l2_norm_eps &&
               grouped_heads == other.grouped_heads;
    }
    void save(BinaryOutputBuffer& buffer) const override {
        primitive_base::save(buffer);
        buffer << history << fuse_qk_l2norm << q_l2_norm_eps << k_l2_norm_eps << grouped_heads;
    }
    void load(BinaryInputBuffer& buffer) override {
        primitive_base::load(buffer);
        buffer >> history >> fuse_qk_l2norm >> q_l2_norm_eps >> k_l2_norm_eps >> grouped_heads;
    }
    int64_t history = 0;
    bool fuse_qk_l2norm = false;
    float q_l2_norm_eps = 1e-6f;
    float k_l2_norm_eps = 1e-6f;
    bool grouped_heads = false;
};
} // namespace cldnn
