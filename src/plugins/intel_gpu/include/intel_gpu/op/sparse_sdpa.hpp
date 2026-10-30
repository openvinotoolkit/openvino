// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief GPU-internal non-paged (dense) sparse-attention consumer for Qwen Sparse Attention (QSA).
///
/// Reads dense K/V history already materialized by an upstream `KVCache`-style producer
/// (read-only; does not write main K/V or own Indexer state), consumes an upstream
/// Indexer's complete-block selection, derives each query's causal tail from its logical
/// position `q_abs = key_length[b] - L + l`, and performs sparse attention over the union.
/// `key_length` describes left padding: logical key `j` of batch `b` is stored at index
/// `S - key_length[b] + j`.
class SparseSDPA : public ov::op::Op {
public:
    OPENVINO_OP("SparseSDPA", "gpu_opset");

    /// @brief Fixed input order. SCALE/SINKS are optional trailing inputs: a node may have
    /// 6 (neither), 7 (SCALE only), or 8 (both) inputs.
    struct InputIdx {
        static constexpr size_t QUERY = 0;                   ///< [B, Hq, L, Dh]
        static constexpr size_t KEY = 1;                     ///< [B, Hkv, S, Dh], dense history
        static constexpr size_t VALUE = 2;                   ///< [B, Hkv, S, Dv], dense history
        static constexpr size_t SELECTED_BLOCK_INDICES = 3;  ///< [B, num_selection_heads, ceil(L/query_block_size), K_max]
        static constexpr size_t SELECTED_BLOCK_COUNT = 4;    ///< [B, num_selection_heads, ceil(L/query_block_size)]
        static constexpr size_t KEY_LENGTH = 5;              ///< [B] i32, valid keys per batch row
        static constexpr size_t SCALE = 6;                   ///< optional scalar or 1-element tensor
        static constexpr size_t SINKS = 7;                   ///< optional [num_heads], requires SCALE

        static constexpr size_t COUNT_MIN = 6;
        static constexpr size_t COUNT_MAX = 8;
    };

    struct Config {
        size_t num_heads = 0;
        size_t num_kv_heads = 0;
        size_t num_selection_heads = 1;  ///< 1, num_kv_heads, or num_heads
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        size_t key_block_size = 0;    ///< matches producer's key_block_size
        size_t query_block_size = 1;  ///< v1 fixed at 1
        bool causal = true;
    };

    SparseSDPA() = default;
    SparseSDPA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }
    ov::element::Type get_output_type() const {
        return m_output_type;
    }

    bool has_scale_input() const {
        return get_input_size() >= InputIdx::SCALE + 1;
    }
    bool has_sinks() const {
        return get_input_size() >= InputIdx::SINKS + 1;
    }

    static bool validate_config(const Config& config, std::string& reason);

private:
    Config m_config{};
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::intel_gpu::op
