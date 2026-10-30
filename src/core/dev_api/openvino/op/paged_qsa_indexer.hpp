// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Paged-cache complete-block plan producer for Qwen Sparse Attention (QSA).
///
/// Consumes externally projected indexer queries/keys (`index_query`/`raw_index_keys`) and the per-token
/// RoPE coefficients of the main attention (`rope_cos`/`rope_sin`), applies query RMSNorm +
/// partial RoPE, appends raw keys, completed-block summaries and block-start RoPE rows into
/// caller-owned paged caches addressed through the shared main-KV page table, and emits
/// per-query selections containing ONLY complete blocks (`selected_block_indices`/`selected_block_count`,
/// width `block_topk`). The causal tail is not represented here; the paired
/// `SparsePA` consumer derives it from each query's logical position. See
/// `paged-qsa-indexer.rst` for the normative contract.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API PagedQSAIndexer : public ov::op::Op {
public:
    OPENVINO_OP("PagedQSAIndexer");

    /// @brief Fixed input order of the PagedQSAIndexer op.
    struct InputIdx {
        static constexpr size_t INDEX_QUERY = 0;           ///< [tokens, H_iq*D_idx]
        static constexpr size_t RAW_INDEX_KEYS = 1;        ///< [tokens, H_ik*D_idx]
        static constexpr size_t ROPE_COS = 2;              ///< [tokens, indexer_rotary_dim]
        static constexpr size_t ROPE_SIN = 3;              ///< [tokens, indexer_rotary_dim]
        static constexpr size_t Q_NORM_WEIGHT = 4;         ///< [D_idx]
        static constexpr size_t K_NORM_WEIGHT = 5;         ///< [D_idx]
        static constexpr size_t PAST_LENS = 6;             ///< [num_sequences]
        static constexpr size_t SUBSEQUENCE_BEGINS = 7;    ///< [num_sequences+1]
        static constexpr size_t BLOCK_INDICES = 8;         ///< [total_logical_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 9;  ///< [num_sequences+1]
        static constexpr size_t INDEX_KEY_CACHE =
            10;  ///< [N_phys, H_ik, page_size, D_idx], writable in place; page_size = shape[2]
        static constexpr size_t INDEX_BLOCK_ROPE_CACHE =
            11;  ///< [N_phys, page_size/r, 2*indexer_rotary_dim], (cos_j, sin_j) interleaved, writable in place
        static constexpr size_t BLOCK_INDEX_KEYS_CACHE = 12;  ///< [N_phys, H_ik, page_size/r, D_idx], writable in place

        static constexpr size_t COUNT = 13;
    };

    struct Config {
        size_t key_block_size = 4;       ///< r
        size_t block_topk = 0;           ///< K_max, required
        size_t num_selection_heads = 1;  ///< 1 or H_ik
        size_t indexer_rotary_dim = 0;   ///< <= indexer_head_dim, even
        size_t indexer_head_dim = 0;     ///< D_idx
        float epsilon = 1e-6f;           ///< RMSNorm epsilon
        float scale = 0.0f;  ///< positive score multiplier; 0 is resolved to 1 / sqrt(indexer_head_dim) by the op
    };

    PagedQSAIndexer() = default;
    PagedQSAIndexer(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

    /// @brief Attribute constraints that must hold before a lowering may be created.
    static bool validate_config(const Config& config, std::string& reason);

    /// `scale`, or its 1 / sqrt(indexer_head_dim) default when unset (0).
    static float effective_scale(const Config& config);

    /// H_iq, derived from index_query's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_query_heads() const;
    /// H_ik, derived from raw_index_keys's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_key_heads() const;

private:
    Config m_config{};
};

}  // namespace ov::op::internal
