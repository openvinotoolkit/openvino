// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief Paged-cache sparse attention consumer primitive: Q0 (K/V cache write) + Q3
/// (selected-block attention with automatically-derived causal tail). Lowered from
/// ov::op::internal::PagedSparseAttention.
struct sparse_pa : public primitive_base<sparse_pa> {
    CLDNN_DECLARE_PRIMITIVE(sparse_pa)

    /// @brief Primitive input order: the leading PagedSparseAttention inputs; scale and sinks are not primitive inputs.
    struct InputIdx {
        static constexpr size_t QUERY = 0;                   ///< [tokens, num_heads*k_head_size]
        static constexpr size_t KEY = 1;                     ///< [tokens, num_kv_heads*k_head_size], current step
        static constexpr size_t VALUE = 2;                   ///< [tokens, num_kv_heads*v_head_size], current step
        static constexpr size_t KEY_CACHE = 3;               ///< [num_blocks, num_kv_heads, page_size, k_head_size], writable
        static constexpr size_t VALUE_CACHE = 4;             ///< [num_blocks, num_kv_heads, page_size, v_head_size], writable
        static constexpr size_t PAST_LENS = 5;               ///< [num_sequences]
        static constexpr size_t SUBSEQUENCE_BEGINS = 6;      ///< [num_sequences+1]
        static constexpr size_t BLOCK_INDICES = 7;           ///< [total_logical_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 8;    ///< [num_sequences+1]
        static constexpr size_t SELECTED_BLOCK_INDICES = 9;  ///< [tokens, num_selection_heads, K_max]
        static constexpr size_t SELECTED_BLOCK_COUNT = 10;   ///< [tokens, num_selection_heads]
        static constexpr size_t COUNT = 11;
    };

    struct Config {
        size_t num_heads = 0;
        size_t num_kv_heads = 0;
        size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        size_t key_block_size = 0;    ///< must divide the page size
        size_t query_block_size = 1;  ///< only 1 is implemented
        bool causal = true;
    };

    static bool validate_config(const Config& c, std::string& reason) {
        if (c.num_heads == 0 || c.num_kv_heads == 0 || c.num_heads % c.num_kv_heads != 0) {
            reason = "num_heads must be a positive multiple of num_kv_heads";
            return false;
        }
        if (c.num_selection_heads != 1 && c.num_selection_heads != c.num_kv_heads && c.num_selection_heads != c.num_heads) {
            reason = "num_selection_heads must be 1, num_kv_heads, or num_heads";
            return false;
        }
        if (c.k_head_size == 0 || c.k_head_size % 16 != 0 || c.v_head_size == 0 || c.v_head_size % 16 != 0) {
            reason = "k_head_size / v_head_size must be positive multiples of 16";
            return false;
        }
        if (c.key_block_size == 0) {
            reason = "key_block_size must be positive";
            return false;
        }
        if (c.query_block_size != 1) {
            reason = "query_block_size != 1 is not implemented in this pass";
            return false;
        }
        return true;
    }

    sparse_pa() : primitive_base("", {}) {}

    /// @param inputs QUERY..SELECTED_BLOCK_COUNT of InputIdx only; scale (a lowered Constant input) is passed resolved.
    sparse_pa(const primitive_id& id, const std::vector<input_info>& inputs, const Config& config, float scale)
        : primitive_base(id, inputs),
          num_heads(config.num_heads),
          num_kv_heads(config.num_kv_heads),
          num_selection_heads(config.num_selection_heads),
          k_head_size(config.k_head_size),
          v_head_size(config.v_head_size),
          key_block_size(config.key_block_size),
          query_block_size(config.query_block_size),
          scale(scale),
          causal(config.causal) {}

    size_t num_heads = 0;
    size_t num_kv_heads = 0;
    size_t num_selection_heads = 1;
    size_t k_head_size = 0;
    size_t v_head_size = 0;
    size_t key_block_size = 0;
    size_t query_block_size = 1;
    float scale = 0.0f;
    bool causal = true;

    Config get_config() const {
        Config config;
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
        seed = hash_combine(seed, scale);
        seed = hash_combine(seed, causal);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto r = downcast<const sparse_pa>(rhs);
        return num_heads == r.num_heads && num_kv_heads == r.num_kv_heads && num_selection_heads == r.num_selection_heads && k_head_size == r.k_head_size &&
               v_head_size == r.v_head_size && key_block_size == r.key_block_size && query_block_size == r.query_block_size && scale == r.scale &&
               causal == r.causal;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<sparse_pa>::save(ob);
        ob << num_heads << num_kv_heads << num_selection_heads << k_head_size << v_head_size << key_block_size << query_block_size << scale << causal;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<sparse_pa>::load(ib);
        ib >> num_heads >> num_kv_heads >> num_selection_heads >> k_head_size >> v_head_size >> key_block_size >> query_block_size >> scale >> causal;
    }
};

}  // namespace cldnn
