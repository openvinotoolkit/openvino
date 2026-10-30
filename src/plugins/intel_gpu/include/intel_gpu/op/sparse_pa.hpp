// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "openvino/op/paged_sparse_attention.hpp"

namespace ov::intel_gpu::op {

/// \brief GPU-internal paged sparse attention consumer: the public ov::op::internal::PagedSparseAttention
/// plus the head geometry the kernels are specialized for (derived from the input shapes by the
/// conversion pass). Lowered to cldnn::sparse_pa.
class SparsePA : public ov::op::internal::PagedSparseAttention {
public:
    OPENVINO_OP("SparsePA", "gpu_opset", ov::op::internal::PagedSparseAttention);

    using Base = ov::op::internal::PagedSparseAttention;

    /// @brief Positional inputs, identical to PagedSparseAttention.
    struct InputIdx {
        static constexpr size_t QUERY = 0;                   ///< [tokens, num_heads*k_head_size]
        static constexpr size_t KEY = 1;                     ///< [tokens, num_kv_heads*k_head_size], current step
        static constexpr size_t VALUE = 2;                   ///< [tokens, num_kv_heads*v_head_size], current step
        static constexpr size_t KEY_CACHE = 3;               ///< [num_blocks, num_kv_heads, page_size, k_head_size], writable
        static constexpr size_t VALUE_CACHE = 4;             ///< [num_blocks, num_kv_heads, page_size, v_head_size], writable
        static constexpr size_t PAST_LENS = 5;               ///< [num_sequences]
        static constexpr size_t SUBSEQUENCE_BEGINS = 6;      ///< [num_sequences+1]
        static constexpr size_t BLOCK_INDICES = 7;           ///< [total_logical_blocks]
        static constexpr size_t BLOCK_INDICES_BEGINS = 8;    ///< [num_sequences+1]
        static constexpr size_t SELECTED_BLOCK_INDICES = 9;  ///< [tokens, num_selection_heads, K_max]
        static constexpr size_t SELECTED_BLOCK_COUNT = 10;   ///< [tokens, num_selection_heads]
        static constexpr size_t SCALE = 11;                  ///< scalar
        static constexpr size_t SINKS = 12;                  ///< [num_heads], or an empty tensor (no sinks)

        static constexpr size_t COUNT = 13;
    };

    struct Config {
        size_t num_heads = 0;
        size_t num_kv_heads = 0;
        size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        size_t key_block_size = 0;    ///< must divide the page size
        size_t query_block_size = 1;  ///< only 1 is implemented
        bool causal = true;
    };

    SparsePA() = default;
    SparsePA(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }

    /// True unless the sinks input is a statically empty tensor.
    bool has_sinks() const;

    static bool validate_config(const Config& config, std::string& reason);

private:
    Config m_config{};
};

}  // namespace ov::intel_gpu::op
