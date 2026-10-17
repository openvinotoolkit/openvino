// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Non-paged, caller-owned-cache complete-block plan producer for Qwen Sparse
/// Attention (QSA). See `qsa-indexer.rst` for the normative contract.
///
/// Reads no page table and owns no opaque runtime state: its logical cache state is two
/// caller-owned writable dense tensors (`summary_cache`, `indexer_raw_k_cache`), mutated
/// in place at logical positions derived from `past_lens`. The caller advances
/// `past_lens` by `L` only after a successful call; this op never resizes either cache
/// or returns replacement cache tensors.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API QSAIndexer : public ov::op::Op {
public:
    OPENVINO_OP("QSAIndexer");

    /// @brief Fixed input order of the QSAIndexer op.
    struct InputIdx {
        static constexpr size_t Q_IDX = 0;                ///< [B, L, H_iq, D_idx]
        static constexpr size_t K_IDX = 1;                ///< [B, L, H_ik, D_idx]
        static constexpr size_t POSITION_IDS = 2;         ///< [B, L]
        static constexpr size_t ROTARY_COS_SIN = 3;       ///< [max_positions, indexer_rotary_dim], interleaved (cos_j, sin_j)
        static constexpr size_t SUMMARY_CACHE = 4;        ///< [B, H_ik, floor(S_cap/r), D_idx], writable in place
        static constexpr size_t INDEXER_RAW_K_CACHE = 5;  ///< [B, H_ik, S_cap, D_idx], writable in place
        static constexpr size_t Q_NORM_WEIGHT = 6;        ///< [D_idx]
        static constexpr size_t K_NORM_WEIGHT = 7;        ///< [D_idx]
        static constexpr size_t PAST_LENS = 8;            ///< [B], read-only
        static constexpr size_t INDEXER_POSITION_CACHE = 9;  ///< [B, floor(S_cap/r)], writable in place

        static constexpr size_t COUNT = 10;
    };

    struct Config {
        size_t compress_ratio = 4;      ///< r
        size_t block_topk = 0;          ///< K_max, required
        size_t h_sel = 1;               ///< 1 or H_ik
        size_t indexer_rotary_dim = 0;  ///< <= indexer_head_dim, even
        size_t indexer_head_dim = 0;    ///< D_idx
        float eps = 1e-6f;              ///< RMSNorm epsilon
        float scale = 0.0f;              ///< 1 / sqrt(indexer_head_dim)
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

    /// H_iq, derived from q_idx's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_query_heads() const;
    /// H_ik, derived from k_idx's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_key_heads() const;

private:
    Config m_config{};
};

}  // namespace ov::op::internal
