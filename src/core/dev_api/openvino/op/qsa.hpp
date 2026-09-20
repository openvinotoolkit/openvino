// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Qwen Sparse Attention (QSA) operation.
///
/// QSA fuses the indexer (block scoring + top-k selection) with the sparse main attention
/// that consumes the selected KV blocks. The op owns four CM kernel families:
///   Q0 qsa_kv_cache_update  - writes the externally projected K_cur/V_cur into the paged f16 cache
///   Q1 qsa_prepare          - indexer Q/K projection, Q RMSNorm + partial RoPE, raw-K append,
///                             completed-block FP32 mean pooling + block-key RMSNorm + block-start RoPE
///   Q2 qsa_score_topk       - block scoring (per-head ReLU before head reduction), top-k
///   Q3 qsa_sparse_attention - selected-block attention with online softmax and output gate
///
/// The op does NOT perform the main Q/K/V projection nor the output projection (`o_proj`);
/// those stay as regular graph operations. Its `mixed` input is the output of
/// GatedResidual READ/MIX, not the raw widened residual.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API QSA : public ov::op::Op {
public:
    OPENVINO_OP("QSA");

    /// @brief Fixed input order of the QSA op / cldnn::qsa primitive.
    struct InputIdx {
        static constexpr size_t QUERY = 0;                   ///< [tokens, Hq*Dh] (or [tokens, 2*Hq*Dh] when gate is fused)
        static constexpr size_t KEY = 1;                     ///< [tokens, Hkv*Dh]
        static constexpr size_t VALUE = 2;                   ///< [tokens, Hkv*Dv]
        static constexpr size_t KEY_CACHE = 3;               ///< [num_blocks, Hkv, PA_BLOCK_SIZE, Dh]
        static constexpr size_t VALUE_CACHE = 4;             ///< [num_blocks, Hkv, PA_BLOCK_SIZE, Dv]
        static constexpr size_t PAST_LENS = 5;               ///< [num_seqs]
        static constexpr size_t SUBSEQUENCE_BEGINS = 6;      ///< [num_seqs+1]
        static constexpr size_t BLOCK_INDICES = 7;           ///< [used_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 8;    ///< [num_seqs+1]
        static constexpr size_t MIXED = 9;                   ///< [tokens, D] GR READ/MIX output
        static constexpr size_t INDEXER_RAW_K_CACHE = 10;    ///< [num_blocks, 1, PA_BLOCK_SIZE, Di]
        static constexpr size_t INDEXER_SUMMARY_CACHE = 11;  ///< [num_blocks, 1, PA_BLOCK_SIZE/r, Di]
        static constexpr size_t INDEXER_QK_WEIGHT = 12;      ///< [Hidx*Di + Di, D] row-major
        static constexpr size_t INDEXER_Q_NORM_WEIGHT = 13;  ///< [Di]
        static constexpr size_t INDEXER_K_NORM_WEIGHT = 14;  ///< [Di]
        static constexpr size_t ROTARY_COS_SIN = 15;         ///< [max_positions, indexer_rotary_dim], (cos,sin) interleaved pairs
        static constexpr size_t POSITION_IDS = 16;           ///< [tokens] absolute position / mRoPE rank

        static constexpr size_t COUNT = 17;
    };

    QSA() = default;

    struct Config {
        size_t heads_num = 0;            ///< Hq
        size_t kv_heads_num = 0;         ///< Hkv
        size_t k_head_size = 0;          ///< Dh
        size_t v_head_size = 0;          ///< Dv
        size_t hidden_size = 0;          ///< D (indexer projection input width)
        size_t indexer_heads = 0;        ///< Hidx
        size_t indexer_kv_heads = 1;     ///< must be 1 (MQA)
        size_t indexer_head_dim = 0;     ///< Di
        size_t indexer_rotary_dim = 0;   ///< <= Di, partial rotary
        size_t budget = 0;               ///< selected token budget
        size_t compress_ratio = 0;       ///< r, per-layer; 0 means the layer is not a QSA layer
        size_t qsa_block_tokens = 0;     ///< == compress_ratio
        size_t pa_block_size = 0;        ///< PA_BLOCK_SIZE of the paged main KV cache
        size_t block_topk = 0;           ///< == budget / compress_ratio
        float scale = 0.0f;              ///< main attention 1/sqrt(Dh)
        float indexer_norm_eps = 1e-6f;  ///< RMSNorm epsilon of the indexer
        bool has_output_gate = false;    ///< q_proj emits [query|gate] per head
        bool is_causal = true;
    };

    QSA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }
    ov::element::Type get_output_type() const {
        return m_output_type;
    }

    /// @brief Attribute constraints that must hold before a QSA primitive may be created.
    /// Returns false (with `reason` filled in) instead of throwing so that callers can fall back.
    static bool validate_config(const Config& config, std::string& reason);

private:
    Config m_config{};
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::op::internal
