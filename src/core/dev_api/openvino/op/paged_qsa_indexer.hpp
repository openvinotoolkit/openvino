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
/// Consumes externally projected indexer queries/keys (`q_idx`/`k_idx`), applies query
/// RMSNorm + partial RoPE, appends raw keys and completed-block summaries into
/// caller-owned paged caches addressed through the shared main-KV page table, and emits
/// per-query selections containing ONLY complete blocks (`sel_indices`/`sel_count`,
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
        static constexpr size_t Q_IDX = 0;                  ///< [tokens, H_iq*D_idx]
        static constexpr size_t K_IDX = 1;                  ///< [tokens, H_ik*D_idx]
        static constexpr size_t POSITION_IDS = 2;           ///< [tokens]
        static constexpr size_t ROTARY_COS_SIN = 3;         ///< [max_positions, indexer_rotary_dim], interleaved (cos_j, sin_j)
        static constexpr size_t SUMMARY_CACHE = 4;          ///< [N_phys, H_ik, page_size/r, D_idx], writable in place
        static constexpr size_t INDEXER_RAW_K_CACHE = 5;    ///< [N_phys, H_ik, page_size, D_idx], writable in place
        static constexpr size_t Q_NORM_WEIGHT = 6;          ///< [D_idx]
        static constexpr size_t K_NORM_WEIGHT = 7;          ///< [D_idx]
        static constexpr size_t BLOCK_INDICES = 8;          ///< [total_logical_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 9;   ///< [num_sequences+1]
        static constexpr size_t SUBSEQUENCE_BEGINS = 10;    ///< [num_sequences+1]
        static constexpr size_t PAST_LENS = 11;             ///< [num_sequences]
        static constexpr size_t INDEXER_POSITION_CACHE = 12;  ///< [N_phys, page_size/r], writable in place

        static constexpr size_t COUNT = 13;
    };

    struct Config {
        size_t compress_ratio = 4;      ///< r
        size_t block_topk = 0;          ///< K_max, required
        size_t h_sel = 1;               ///< 1 or H_ik
        size_t indexer_rotary_dim = 0;  ///< <= indexer_head_dim, even
        size_t indexer_head_dim = 0;    ///< D_idx
        size_t page_size = 0;           ///< shared main-KV page size; must be a multiple of compress_ratio
        float eps = 1e-6f;              ///< RMSNorm epsilon
        float scale = 0.0f;             ///< 1 / sqrt(indexer_head_dim)
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

    /// H_iq, derived from q_idx's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_query_heads() const;
    /// H_ik, derived from k_idx's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_key_heads() const;

private:
    Config m_config{};
};

}  // namespace ov::op::internal
