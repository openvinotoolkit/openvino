// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/op/sparse_pa.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief Paged-cache sparse attention consumer primitive: Q0 (K/V cache write) + Q3
/// (selected-block attention with automatically-derived causal tail). See
/// ov::intel_gpu::op::SparsePA.
struct sparse_pa : public primitive_base<sparse_pa> {
    CLDNN_DECLARE_PRIMITIVE(sparse_pa)

    using InputIdx = ov::intel_gpu::op::SparsePA::InputIdx;

    sparse_pa() : primitive_base("", {}) {}

    sparse_pa(const primitive_id& id, const std::vector<input_info>& inputs, const ov::intel_gpu::op::SparsePA::Config& config)
        : primitive_base(id, inputs),
          num_heads(config.num_heads),
          num_kv_heads(config.num_kv_heads),
          h_sel(config.h_sel),
          k_head_size(config.k_head_size),
          v_head_size(config.v_head_size),
          block_size(config.block_size),
          sel_block_size(config.sel_block_size),
          q_block_size(config.q_block_size),
          scale(config.scale),
          causal(config.causal) {}

    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t h_sel = 1;
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t block_size = 0;
    size_t sel_block_size = 0;
    size_t q_block_size = 1;
    float scale = 0.0f;
    bool causal = true;

    ov::intel_gpu::op::SparsePA::Config get_config() const {
        ov::intel_gpu::op::SparsePA::Config config;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.h_sel = h_sel;
        config.k_head_size = k_head_size;
        config.v_head_size = v_head_size;
        config.block_size = block_size;
        config.sel_block_size = sel_block_size;
        config.q_block_size = q_block_size;
        config.scale = scale;
        config.causal = causal;
        return config;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_heads);
        seed = hash_combine(seed, num_kv_heads);
        seed = hash_combine(seed, h_sel);
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, block_size);
        seed = hash_combine(seed, sel_block_size);
        seed = hash_combine(seed, q_block_size);
        seed = hash_combine(seed, scale);
        seed = hash_combine(seed, causal);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto r = downcast<const sparse_pa>(rhs);
        return num_heads == r.num_heads && num_kv_heads == r.num_kv_heads && h_sel == r.h_sel && k_head_size == r.k_head_size &&
               v_head_size == r.v_head_size && block_size == r.block_size && sel_block_size == r.sel_block_size && q_block_size == r.q_block_size &&
               scale == r.scale && causal == r.causal;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<sparse_pa>::save(ob);
        ob << num_heads << num_kv_heads << h_sel << k_head_size << v_head_size << block_size << sel_block_size << q_block_size << scale << causal;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<sparse_pa>::load(ib);
        ib >> num_heads >> num_kv_heads >> h_sel >> k_head_size >> v_head_size >> block_size >> sel_block_size >> q_block_size >> scale >> causal;
    }
};

}  // namespace cldnn
