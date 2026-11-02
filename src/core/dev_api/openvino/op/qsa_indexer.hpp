// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Non-paged, stateful complete-block plan producer for Qwen Sparse Attention (QSA).
///
/// Fuses opset17 QSAPrepare and SparseAttentionIndexer. The state (`block_index_keys_cache`, and the
/// raw keys and first-token RoPE row of the incomplete block) is passed as inputs and returned,
/// updated, as outputs. `block_index_keys_cache` must already hold the complete blocks of the longest
/// sequence after the call, so its shape is not changed. `key_length` is the number of valid tokens of
/// each row after the call. RoPE is the rotate-half RoPE of the first `indexer_rotary_dim` channels with
/// the per-token `rope_cos` / `rope_sin`.
///
/// The inputs, outputs and attributes match the GPU plugin QSAIndexer op. The raw keys are RMS-normalized
/// with `epsilon` and `k_norm_weight`, and then rotated. The query gets the same steps when `query_norm` and
/// `query_rope` are set (the GPU contract); otherwise `index_query` already holds the normalized or rotated
/// query and `q_norm_weight` is unused. The query is normalized before it is rotated.
class OPENVINO_API QSAIndexer : public ov::op::Op {
public:
    OPENVINO_OP("QSAIndexer");

    /// @brief Fixed input order of the QSAIndexer op.
    struct InputIdx {
        static constexpr size_t INDEX_QUERY = 0;             ///< [B, L, H_iq, D_idx]
        static constexpr size_t RAW_INDEX_KEYS = 1;          ///< [B, L, H_ik, D_idx]
        static constexpr size_t ROPE_COS = 2;                ///< [B, L, indexer_rotary_dim]
        static constexpr size_t ROPE_SIN = 3;                ///< [B, L, indexer_rotary_dim]
        static constexpr size_t KEY_LENGTH = 4;              ///< [B], i32, valid tokens after the call
        static constexpr size_t Q_NORM_WEIGHT = 5;           ///< [D_idx]
        static constexpr size_t K_NORM_WEIGHT = 6;           ///< [D_idx]
        static constexpr size_t BLOCK_INDEX_KEYS_CACHE = 7;  ///< [B, H_ik, blocks, D_idx]
        static constexpr size_t PAST_INCOMPLETE_RAW_K = 8;   ///< [B, H_ik, r - 1, D_idx]
        static constexpr size_t PAST_INCOMPLETE_ROPE = 9;    ///< [B, 2 * indexer_rotary_dim], (cos_j, sin_j)

        static constexpr size_t COUNT = 10;
    };

    /// @brief Fixed output order of the QSAIndexer op.
    struct OutputIdx {
        static constexpr size_t SELECTED_BLOCK_INDICES = 0;  ///< [B, num_selection_heads, L, block_topk]
        static constexpr size_t SELECTED_BLOCK_COUNT = 1;    ///< [B, num_selection_heads, L]
        static constexpr size_t BLOCK_INDEX_KEYS_CACHE = 2;  ///< same shape as input BLOCK_INDEX_KEYS_CACHE
        static constexpr size_t PAST_INCOMPLETE_RAW_K = 3;   ///< same shape as input PAST_INCOMPLETE_RAW_K
        static constexpr size_t PAST_INCOMPLETE_ROPE = 4;    ///< same shape as input PAST_INCOMPLETE_ROPE

        static constexpr size_t COUNT = 5;
    };

    struct Config {
        size_t key_block_size = 4;       ///< r
        size_t block_topk = 0;           ///< K_max, required
        size_t num_selection_heads = 1;  ///< 1 or H_ik
        size_t indexer_rotary_dim = 0;   ///< <= indexer_head_dim, even
        size_t indexer_head_dim = 0;     ///< D_idx
        float epsilon = 1e-6f;           ///< query and key RMSNorm epsilon
        float scale = 0.0f;      ///< positive score multiplier; 0 selects the default 1 / sqrt(indexer_head_dim)
        bool query_norm = true;  ///< RMS-normalize index_query with epsilon and q_norm_weight
        bool query_rope = true;  ///< rotate index_query with rope_cos / rope_sin
    };

    QSAIndexer() = default;
    QSAIndexer(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

    /// @brief Attribute constraints that must hold before the op may be created.
    static bool validate_config(const Config& config, std::string& reason);

    /// `scale`, or its 1 / sqrt(indexer_head_dim) default when unset (0).
    static float effective_scale(const Config& config);

private:
    Config m_config{};
};

}  // namespace ov::op::internal
