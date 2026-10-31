// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/op/sparse_sdpa.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief Non-paged (dense) sparse attention consumer primitive: Q3 only, read-only on
/// main K/V. See ov::intel_gpu::op::SparseSDPA. Inputs: query, key, value, selected_block_indices,
/// selected_block_count, key_length. A constant scale input is folded into `scale`; sinks are not
/// supported, lowering must reject them explicitly rather than construct this primitive.
struct sparse_sdpa : public primitive_base<sparse_sdpa> {
    CLDNN_DECLARE_PRIMITIVE(sparse_sdpa)

    using InputIdx = ov::intel_gpu::op::SparseSDPA::InputIdx;

    sparse_sdpa() : primitive_base("", {}) {}

    sparse_sdpa(const primitive_id& id, const std::vector<input_info>& inputs, const ov::intel_gpu::op::SparseSDPA::Config& config, float resolved_scale)
        : primitive_base(id, inputs),
          num_heads(config.num_heads),
          num_kv_heads(config.num_kv_heads),
          num_selection_heads(config.num_selection_heads),
          k_head_size(config.k_head_size),
          v_head_size(config.v_head_size),
          key_block_size(config.key_block_size),
          query_block_size(config.query_block_size),
          causal(config.causal),
          scale(resolved_scale) {}

    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t num_selection_heads = 1;
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t key_block_size = 0;
    size_t query_block_size = 1;
    bool causal = true;
    /// Resolved at lowering time: either the op's default (1/sqrt(k_head_size)) or a
    /// constant-folded value of the optional SCALE input.
    float scale = 0.0f;

    ov::intel_gpu::op::SparseSDPA::Config get_config() const {
        ov::intel_gpu::op::SparseSDPA::Config config;
        config.num_heads = num_heads;
        config.num_kv_heads = num_kv_heads;
        config.num_selection_heads = num_selection_heads;
        config.k_head_size = k_head_size;
        config.v_head_size = v_head_size;
        config.key_block_size = key_block_size;
        config.query_block_size = query_block_size;
        config.causal = causal;
        return config;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, num_heads);
        seed = hash_combine(seed, num_kv_heads);
        seed = hash_combine(seed, num_selection_heads);
        seed = hash_combine(seed, k_head_size);
        seed = hash_combine(seed, v_head_size);
        seed = hash_combine(seed, key_block_size);
        seed = hash_combine(seed, query_block_size);
        seed = hash_combine(seed, causal);
        seed = hash_combine(seed, scale);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto r = downcast<const sparse_sdpa>(rhs);
        return num_heads == r.num_heads && num_kv_heads == r.num_kv_heads && num_selection_heads == r.num_selection_heads && k_head_size == r.k_head_size &&
               v_head_size == r.v_head_size && key_block_size == r.key_block_size && query_block_size == r.query_block_size && causal == r.causal &&
               scale == r.scale;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<sparse_sdpa>::save(ob);
        ob << num_heads << num_kv_heads << num_selection_heads << k_head_size << v_head_size << key_block_size << query_block_size << causal << scale;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<sparse_sdpa>::load(ib);
        ib >> num_heads >> num_kv_heads >> num_selection_heads >> k_head_size >> v_head_size >> key_block_size >> query_block_size >> causal >> scale;
    }
};

}  // namespace cldnn
