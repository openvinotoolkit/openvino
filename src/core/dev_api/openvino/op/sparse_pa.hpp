// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Paged-cache sparse-attention consumer for Qwen Sparse Attention (QSA).
///
/// Writes the current-step K/V into the caller-owned paged main KV cache (Q0), consumes
/// an upstream Indexer's complete-block selection, derives each query's causal tail from
/// its logical position, and performs sparse attention over the union (Q3). v1 only
/// implements the `sel_indices`/`sel_count` plan form; `sel_block_mask` is not part of
/// this op's ABI in this pass (see `sparse-pa.rst` for the full, future mask-only form).
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SparsePA : public ov::op::Op {
public:
    OPENVINO_OP("SparsePA");

    /// @brief Fixed input order of the SparsePA op (indices/count plan form only).
    struct InputIdx {
        static constexpr size_t QUERY = 0;                 ///< [tokens, num_heads*k_head_size]
        static constexpr size_t KEY = 1;                   ///< [tokens, num_kv_heads*k_head_size], current step
        static constexpr size_t VALUE = 2;                 ///< [tokens, num_kv_heads*v_head_size], current step
        static constexpr size_t KEY_CACHE = 3;             ///< [num_blocks, num_kv_heads, block_size, k_head_size], writable
        static constexpr size_t VALUE_CACHE = 4;           ///< [num_blocks, num_kv_heads, block_size, v_head_size], writable
        static constexpr size_t BLOCK_INDICES = 5;         ///< [total_logical_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 6;  ///< [num_sequences+1]
        static constexpr size_t SUBSEQUENCE_BEGINS = 7;    ///< [num_sequences+1]
        static constexpr size_t PAST_LENS = 8;             ///< [num_sequences]
        static constexpr size_t SEL_INDICES = 9;           ///< [total_q_blocks, h_sel, K_max]
        static constexpr size_t SEL_COUNT = 10;            ///< [total_q_blocks, h_sel]
        static constexpr size_t QB_START = 11;             ///< [num_sequences+1]

        static constexpr size_t COUNT = 12;
    };

    struct Config {
        size_t num_heads = 0;
        size_t num_kv_heads = 0;
        size_t h_sel = 1;            ///< 1, num_kv_heads, or num_heads
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        size_t block_size = 0;       ///< PA_BLOCK_SIZE
        size_t sel_block_size = 0;   ///< matches producer's compress_ratio; must divide block_size
        size_t q_block_size = 1;     ///< v1 fixed at 1
        float scale = 0.0f;
        bool causal = true;
    };

    SparsePA() = default;
    SparsePA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }
    ov::element::Type get_output_type() const {
        return m_output_type;
    }

    static bool validate_config(const Config& config, std::string& reason);

private:
    Config m_config{};
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::op::internal
