// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/op/op.hpp"
#include "transformations_visibility.hpp"

namespace ov::op::internal {

/// Graph-only placeholder for the paged QSA indexer; it has no execution implementation.
class TRANSFORMATIONS_API PagedQSAIndexer : public ov::op::Op {
public:
    OPENVINO_OP("PagedQSAIndexer");

    struct InputIdx {
        static constexpr size_t INDEX_QUERY = 0;              ///< [tokens, H_iq * D_idx]
        static constexpr size_t RAW_INDEX_KEYS = 1;           ///< [tokens, H_ik * D_idx]
        static constexpr size_t ROPE_COS = 2;                 ///< [tokens, indexer_rotary_dim]
        static constexpr size_t ROPE_SIN = 3;                 ///< [tokens, indexer_rotary_dim]
        static constexpr size_t Q_NORM_WEIGHT = 4;            ///< [D_idx]
        static constexpr size_t K_NORM_WEIGHT = 5;            ///< [D_idx]
        static constexpr size_t PAST_LENS = 6;                ///< [num_sequences]
        static constexpr size_t SUBSEQUENCE_BEGINS = 7;       ///< [num_sequences + 1]
        static constexpr size_t BLOCK_INDICES = 8;            ///< [total_logical_pages]
        static constexpr size_t BLOCK_INDICES_BEGINS = 9;     ///< [num_sequences + 1]
        static constexpr size_t INDEX_KEY_CACHE = 10;         ///< [pages, H_ik, page_size, D_idx]
        static constexpr size_t INDEX_BLOCK_ROPE_CACHE = 11;  ///< [pages, page_size / r, 2 * rotary_dim]
        static constexpr size_t BLOCK_INDEX_KEYS_CACHE = 12;  ///< [pages, H_ik, page_size / r, D_idx]
        static constexpr size_t COUNT = 13;
    };

    struct OutputIdx {
        static constexpr size_t SELECTED_BLOCK_INDICES = 0;  ///< [tokens, selection_heads, block_topk]
        static constexpr size_t SELECTED_BLOCK_COUNT = 1;    ///< [tokens, selection_heads]
        static constexpr size_t COUNT = 2;
    };

    struct Config {
        size_t key_block_size = 4;
        size_t block_topk = 0;
        size_t num_selection_heads = 1;
        size_t indexer_rotary_dim = 0;
        size_t indexer_head_dim = 0;
        float epsilon = 1e-6f;
        float scale = 0.0f;      ///< 0 selects 1 / sqrt(indexer_head_dim), resolved during validation.
        bool query_norm = true;  ///< False when query RMSNorm remains outside the operation.
        bool query_rope = true;  ///< False when query RoPE remains outside the operation.
    };

    PagedQSAIndexer() = default;
    PagedQSAIndexer(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

private:
    Config m_config{};
};

}  // namespace ov::op::internal
