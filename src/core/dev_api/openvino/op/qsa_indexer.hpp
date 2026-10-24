// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Non-paged, stateful complete-block plan producer for Qwen Sparse Attention (QSA).
/// See `qsa-indexer.rst` for the normative contract.
///
/// Reads no page table and owns no opaque runtime state. The state (`block_index_keys_cache`, and the
/// raw keys and first-token RoPE row of the incomplete block) is passed as inputs and
/// returned, updated, as outputs (functional state interface). `key_length` is the number of
/// valid tokens of each row after the call; rows shorter than `L` are left-padded.
/// RoPE uses the caller-supplied per-token rotate-half `rope_cos`/`rope_sin`.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API QSAIndexer : public ov::op::Op {
public:
    OPENVINO_OP("QSAIndexer");

    /// @brief Fixed input order of the QSAIndexer op.
    struct InputIdx {
        static constexpr size_t INDEX_QUERY = 0;                  ///< [B, L, H_iq, D_idx]
        static constexpr size_t RAW_INDEX_KEYS = 1;                  ///< [B, L, H_ik, D_idx]
        static constexpr size_t ROPE_COS = 2;               ///< [B, L, indexer_rotary_dim]
        static constexpr size_t ROPE_SIN = 3;               ///< [B, L, indexer_rotary_dim]
        static constexpr size_t KEY_LENGTH = 4;             ///< [B], valid tokens after the call
        static constexpr size_t Q_NORM_WEIGHT = 5;          ///< [D_idx]
        static constexpr size_t K_NORM_WEIGHT = 6;          ///< [D_idx]
        static constexpr size_t BLOCK_INDEX_KEYS_CACHE = 7;          ///< [B, H_ik, floor(S_cap/r), D_idx]
        static constexpr size_t PAST_INCOMPLETE_RAW_K = 8;  ///< [B, H_ik, r-1, D_idx], slot p % r
        static constexpr size_t PAST_INCOMPLETE_ROPE = 9;   ///< [B, 2*indexer_rotary_dim], (cos_j, sin_j) interleaved

        static constexpr size_t COUNT = 10;
    };

    /// @brief Fixed output order of the QSAIndexer op.
    struct OutputIdx {
        static constexpr size_t SELECTED_BLOCK_INDICES = 0;            ///< [B, num_selection_heads, L, block_topk]
        static constexpr size_t SELECTED_BLOCK_COUNT = 1;              ///< [B, num_selection_heads, L]
        static constexpr size_t BLOCK_INDEX_KEYS_CACHE = 2;          ///< same shape as input BLOCK_INDEX_KEYS_CACHE
        static constexpr size_t PAST_INCOMPLETE_RAW_K = 3;  ///< same shape as input PAST_INCOMPLETE_RAW_K
        static constexpr size_t PAST_INCOMPLETE_ROPE = 4;   ///< same shape as input PAST_INCOMPLETE_ROPE

        static constexpr size_t COUNT = 5;
    };

    struct Config {
        size_t key_block_size = 4;      ///< r
        size_t block_topk = 0;          ///< K_max, required
        size_t num_selection_heads = 1;               ///< 1 or H_ik
        size_t indexer_rotary_dim = 0;  ///< <= indexer_head_dim, even
        size_t indexer_head_dim = 0;    ///< D_idx
        float epsilon = 1e-6f;              ///< RMSNorm epsilon
        float scale = 0.0f;             ///< positive score multiplier; 0 selects the default 1 / sqrt(indexer_head_dim)
    };

    QSAIndexer() = default;
    QSAIndexer(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

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
