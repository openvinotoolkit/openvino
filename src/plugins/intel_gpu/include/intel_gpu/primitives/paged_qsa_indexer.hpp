// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "intel_gpu/op/paged_qsa_indexer.hpp"
#include "primitive.hpp"

namespace cldnn {

/// @brief Paged-cache QSA Indexer primitive: Q1 (prepare) + Q2 (score/top-k).
/// @details See ov::intel_gpu::op::PagedQSAIndexer. Always declares 2 outputs
/// (sel_indices, sel_count); the raw-K/summary cache inputs are mutated in place.
struct paged_qsa_indexer : public primitive_base<paged_qsa_indexer> {
    CLDNN_DECLARE_PRIMITIVE(paged_qsa_indexer)

    using InputIdx = ov::intel_gpu::op::PagedQSAIndexer::InputIdx;

    paged_qsa_indexer() : primitive_base("", {}) {}

    paged_qsa_indexer(const primitive_id& id, const std::vector<input_info>& inputs, const ov::intel_gpu::op::PagedQSAIndexer::Config& config)
        : primitive_base(id, inputs, 2, {optional_data_type{ov::element::i32}, optional_data_type{ov::element::i32}}),
          compress_ratio(config.compress_ratio),
          block_topk(config.block_topk),
          h_sel(config.h_sel),
          indexer_rotary_dim(config.indexer_rotary_dim),
          indexer_head_dim(config.indexer_head_dim),
          page_size(config.page_size),
          eps(config.eps),
          scale(config.scale) {}

    size_t compress_ratio = 0;
    size_t block_topk = 0;
    size_t h_sel = 1;
    size_t indexer_rotary_dim = 0;
    size_t indexer_head_dim = 0;
    size_t page_size = 0;
    float eps = 1e-6f;
    float scale = 0.0f;

    ov::intel_gpu::op::PagedQSAIndexer::Config get_config() const {
        ov::intel_gpu::op::PagedQSAIndexer::Config config;
        config.compress_ratio = compress_ratio;
        config.block_topk = block_topk;
        config.h_sel = h_sel;
        config.indexer_rotary_dim = indexer_rotary_dim;
        config.indexer_head_dim = indexer_head_dim;
        config.page_size = page_size;
        config.eps = eps;
        config.scale = scale;
        return config;
    }

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, compress_ratio);
        seed = hash_combine(seed, block_topk);
        seed = hash_combine(seed, h_sel);
        seed = hash_combine(seed, indexer_rotary_dim);
        seed = hash_combine(seed, indexer_head_dim);
        seed = hash_combine(seed, page_size);
        seed = hash_combine(seed, scale);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        auto r = downcast<const paged_qsa_indexer>(rhs);
        return compress_ratio == r.compress_ratio && block_topk == r.block_topk && h_sel == r.h_sel && indexer_rotary_dim == r.indexer_rotary_dim &&
               indexer_head_dim == r.indexer_head_dim && page_size == r.page_size && scale == r.scale;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<paged_qsa_indexer>::save(ob);
        ob << compress_ratio << block_topk << h_sel << indexer_rotary_dim << indexer_head_dim << page_size << eps << scale;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<paged_qsa_indexer>::load(ib);
        ib >> compress_ratio >> block_topk >> h_sel >> indexer_rotary_dim >> indexer_head_dim >> page_size >> eps >> scale;
    }
};

}  // namespace cldnn
