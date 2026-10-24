// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief GPU-internal paged-cache sparse-attention consumer for Qwen Sparse Attention (QSA).
///
/// Writes the current-step K/V into the caller-owned paged main KV cache (Q0), consumes
/// an upstream Indexer's complete-block selection, derives each query's causal tail from
/// its logical position, and performs sparse attention over the union (Q3). v1 only
/// implements the `selected_block_indices`/`selected_block_count` plan form; `selected_block_mask` (the future
/// mask-only form) is an optional trailing input that the GPU lowering rejects.
/// `scale` is a scalar input and `sinks` an input that must be empty (sinks are not supported).
/// `qb_start` is an optional trailing input: only query_block_size == 1 is implemented, where it equals
/// `subsequence_begins`. The op has 13..15 inputs; an empty tensor stands for an absent optional input.
class SparsePA : public ov::op::Op {
public:
    OPENVINO_OP("SparsePA", "gpu_opset");

    /// @brief Fixed input order of the SparsePA op (indices/count plan form only).
    struct InputIdx {
        static constexpr size_t QUERY = 0;                   ///< [tokens, num_heads*k_head_size]
        static constexpr size_t KEY = 1;                     ///< [tokens, num_kv_heads*k_head_size], current step
        static constexpr size_t VALUE = 2;                   ///< [tokens, num_kv_heads*v_head_size], current step
        static constexpr size_t KEY_CACHE = 3;               ///< [num_blocks, num_kv_heads, page_size, k_head_size], writable; page_size = shape[2]
        static constexpr size_t VALUE_CACHE = 4;             ///< [num_blocks, num_kv_heads, page_size, v_head_size], writable
        static constexpr size_t PAST_LENS = 5;               ///< [num_sequences]
        static constexpr size_t SUBSEQUENCE_BEGINS = 6;      ///< [num_sequences+1]
        static constexpr size_t BLOCK_INDICES = 7;           ///< [total_logical_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 8;    ///< [num_sequences+1]
        static constexpr size_t SELECTED_BLOCK_INDICES = 9;  ///< [total_q_blocks, num_selection_heads, K_max]
        static constexpr size_t SELECTED_BLOCK_COUNT = 10;   ///< [total_q_blocks, num_selection_heads]
        static constexpr size_t SCALE = 11;                  ///< scalar
        static constexpr size_t SINKS = 12;                  ///< [num_heads], or an empty tensor (no sinks)
        static constexpr size_t QB_START = 13;               ///< optional [num_sequences+1]; absent means subsequence_begins (query_block_size == 1)

        static constexpr size_t SELECTED_BLOCK_MASK = 14;  ///< optional bool [total_q_blocks, num_selection_heads, max_k_blocks]

        static constexpr size_t COUNT_MIN = 13;  ///< without QB_START and SELECTED_BLOCK_MASK
        static constexpr size_t COUNT_MAX = 15;
    };

    struct Config {
        size_t num_heads = 0;
        size_t num_kv_heads = 0;
        size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        size_t key_block_size = 0;    ///< matches producer's key_block_size; must divide the page size
        size_t query_block_size = 1;  ///< v1 fixed at 1
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

    /// True unless the sinks input is a statically empty tensor.
    bool has_sinks() const;

    /// True when the optional selected_block_mask input is present and not statically empty.
    bool has_selected_block_mask() const;

    static bool validate_config(const Config& config, std::string& reason);

private:
    Config m_config{};
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::intel_gpu::op
