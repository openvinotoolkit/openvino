// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <memory>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

/// \brief Padding-aware, non-paged, caller-owned-cache complete-block plan producer for Qwen
/// Sparse Attention (QSA). See `qsa-indexer-v2.rst` for the normative contract.
///
/// Unlike `QSAIndexer`, every B x L query/key token need not be real: `key_length` tells this op,
/// per row, the total number of valid (past + current) key tokens *after* this call, and any
/// leading rows of this call's L-wide window beyond that count are treated as left-padding and
/// skipped (no pooling contribution, no selection). `past`/`current`/block-append-offset are all
/// derived per row from `key_length` and `L` alone -- the caller advances no running counter of
/// its own between calls, unlike `QSAIndexer`'s `past_lens`.
///
/// Reads no page table and owns no opaque runtime state: its logical cache state is three
/// caller-owned dense tensors (`summary_cache`, `past_incomplete_raw_k`, `past_incomplete_rope`).
/// The update is NOT in place: the inputs are never modified, and the updated values of the three
/// states are produced as outputs (same shape/type as the matching input) so that an outside
/// Assign can write them back to the variables. `summary_cache` grows across calls
/// (caller-managed, batch-max sized); `past_incomplete_raw_k`/`past_incomplete_rope` are FIXED size
/// (`key_block_size - 1` rows / one row respectively) and fully overwritten every call.
///
/// \ingroup ov_ops_cpp_api
class QSAIndexer : public ov::op::Op {
public:
    OPENVINO_OP("QSAIndexer", "cpu_plugin_opset");

    /// @brief Fixed input order of the QSAIndexer op. All state cache ports are grouped at
    /// the end, after every plain-data/weight input.
    struct InputIdx {
        static constexpr size_t Q_IDX = 0;          ///< [B, L, H_iq, D_idx]
        static constexpr size_t K_IDX = 1;          ///< [B, L, H_ik, D_idx], may include left-padding rows
        static constexpr size_t ROPE_COS = 2;       ///< [B or 1, L, indexer_rotary_dim]
        static constexpr size_t ROPE_SIN = 3;       ///< [B or 1, L, indexer_rotary_dim]
        static constexpr size_t KEY_LENGTH = 4;     ///< [B] i32, read-only; total valid length after this call
        static constexpr size_t Q_NORM_WEIGHT = 5;  ///< [D_idx]
        static constexpr size_t K_NORM_WEIGHT = 6;  ///< [D_idx]
        static constexpr size_t SUMMARY_CACHE = 7;  ///< [B, H_ik, Bp, D_idx], state input
        static constexpr size_t PAST_INCOMPLETE_RAW_K =
            8;  ///< [B, H_ik, key_block_size - 1, D_idx], state input, fixed size
        static constexpr size_t PAST_INCOMPLETE_ROPE =
            9;  ///< [B, 2 * indexer_rotary_dim], rope_cos then rope_sin of the incomplete block's first token, state
                ///< input, fixed size

        static constexpr size_t COUNT = 10;
    };

    /// @brief Fixed output order of the QSAIndexer op. The three state outputs are the updated
    /// values of the matching state inputs (same shape and element type).
    struct OutputIdx {
        static constexpr size_t SEL_INDICES = 0;            ///< [B, H_sel, L, block_topk] i32
        static constexpr size_t SEL_COUNT = 1;              ///< [B, H_sel, L] i32
        static constexpr size_t SUMMARY_CACHE = 2;          ///< updated summary_cache
        static constexpr size_t PAST_INCOMPLETE_RAW_K = 3;  ///< updated past_incomplete_raw_k
        static constexpr size_t PAST_INCOMPLETE_ROPE = 4;   ///< updated past_incomplete_rope

        static constexpr size_t COUNT = 5;
    };

    struct Config {
        size_t compress_ratio = 4;      ///< r (key_block_size)
        size_t block_topk = 0;          ///< K_max, required
        size_t h_sel = 1;               ///< 1 or H_ik
        size_t indexer_rotary_dim = 0;  ///< <= indexer_head_dim, even
        size_t indexer_head_dim = 0;    ///< D_idx
        float eps = 1e-6f;              ///< RMSNorm epsilon
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

    /// H_iq, derived from q_idx's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_query_heads() const;
    /// H_ik, derived from k_idx's static shape (last dim / indexer_head_dim). Never cached.
    size_t get_index_key_heads() const;

private:
    Config m_config{};
};

}  // namespace ov::intel_cpu
