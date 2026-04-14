// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>

#include "primitive.hpp"

namespace cldnn {

struct pa_kv_reorder : public primitive_base<pa_kv_reorder> {
    CLDNN_DECLARE_PRIMITIVE(pa_kv_reorder)

    enum PaKVReorderInputIdx {
        KEY_CACHE = 0,
        VALUE_CACHE = 1,
        BLOCK_INDICES = 2,
        BLOCK_INDICES_BEGINS = 3,
        BLOCK_UPDATE_INDICES = 4,
        BLOCK_UPDATE_INDICES_BEGINS = 5,
    };

    pa_kv_reorder() : primitive_base("", {}, 1) {}

    pa_kv_reorder(primitive_id id, std::vector<input_info> inputs) : primitive_base(std::move(id), std::move(inputs), 1) {
        OPENVINO_ASSERT(input.size() == 6, "[GPU] Unexpected input number for pa_kv_reorder primitive: ", input.size());
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, is_key_by_channel);
        seed = hash_combine(seed, scales_zp_size);
        seed = hash_combine(seed, kv_heads_num);
        seed = hash_combine(seed, adjusted_k_head_size);
        seed = hash_combine(seed, adjusted_paged_attention_block_size);
        seed = hash_combine(seed, adjusted_v_head_size);
        seed = hash_combine(seed, static_cast<size_t>(cache_dt));
        seed = hash_combine(seed, is_kv_compressed);
        for (const auto value : key_cache_dim_order) {
            seed = hash_combine(seed, value);
        }
        for (const auto value : value_cache_dim_order) {
            seed = hash_combine(seed, value);
        }
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const pa_kv_reorder>(rhs);
        return is_key_by_channel == rhs_casted.is_key_by_channel && scales_zp_size == rhs_casted.scales_zp_size && kv_heads_num == rhs_casted.kv_heads_num &&
               adjusted_k_head_size == rhs_casted.adjusted_k_head_size &&
               adjusted_paged_attention_block_size == rhs_casted.adjusted_paged_attention_block_size &&
               adjusted_v_head_size == rhs_casted.adjusted_v_head_size && cache_dt == rhs_casted.cache_dt && is_kv_compressed == rhs_casted.is_kv_compressed &&
               key_cache_dim_order == rhs_casted.key_cache_dim_order && value_cache_dim_order == rhs_casted.value_cache_dim_order;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<pa_kv_reorder>::save(ob);
        ob << is_key_by_channel;
        ob << scales_zp_size;
        ob << kv_heads_num;
        ob << adjusted_k_head_size;
        ob << adjusted_paged_attention_block_size;
        ob << adjusted_v_head_size;
        ob << static_cast<int64_t>(cache_dt);
        ob << is_kv_compressed;
        for (const auto value : key_cache_dim_order) {
            ob << value;
        }
        for (const auto value : value_cache_dim_order) {
            ob << value;
        }
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<pa_kv_reorder>::load(ib);
        ib >> is_key_by_channel;
        ib >> scales_zp_size;
        ib >> kv_heads_num;
        ib >> adjusted_k_head_size;
        ib >> adjusted_paged_attention_block_size;
        ib >> adjusted_v_head_size;
        int64_t cache_dt_val = static_cast<int64_t>(data_types::f16);
        ib >> cache_dt_val;
        cache_dt = static_cast<data_types>(cache_dt_val);
        ib >> is_kv_compressed;
        for (auto& value : key_cache_dim_order) {
            ib >> value;
        }
        for (auto& value : value_cache_dim_order) {
            ib >> value;
        }
    }

    bool is_key_by_channel = false;
    size_t scales_zp_size = 0;
    size_t kv_heads_num = 0;
    size_t adjusted_k_head_size = 0;
    size_t adjusted_paged_attention_block_size = 0;
    size_t adjusted_v_head_size = 0;
    data_types cache_dt = data_types::f16;
    bool is_kv_compressed = false;
    std::array<size_t, 4> key_cache_dim_order = {0, 1, 3, 2};
    std::array<size_t, 4> value_cache_dim_order = {0, 1, 2, 3};
};

}  // namespace cldnn
