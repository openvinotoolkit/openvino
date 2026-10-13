// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <string>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Non-paged (dense) sparse-attention consumer for Qwen Sparse Attention (QSA).
///
/// Reads dense K/V history already materialized by an upstream `KVCache`-style producer
/// (read-only; does not write main K/V or own Indexer state), consumes an upstream
/// Indexer's complete-block selection, derives each query's causal tail using the
/// bottom-right-aligned logical position `q_abs = S - L + l`, and performs sparse
/// attention over the union. v1 only implements the `sel_indices`/`sel_count` plan form
/// (no plan-carrying `attn_mask`); an ordinary, non-plan `attn_mask` is not implemented
/// in GPU lowering in this pass either (explicit rejection, no silent fallback) -- see
/// `sparse-sdpa.rst` for the full normative contract.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API SparseSDPA : public ov::op::Op {
public:
    OPENVINO_OP("SparseSDPA");

    /// @brief Fixed input order. ATTN_MASK/SCALE are optional trailing inputs: a node may
    /// have 5 (neither), 6 (ATTN_MASK only), or 7 (both) inputs.
    struct InputIdx {
        static constexpr size_t QUERY = 0;        ///< [B, Hq, L, Dh]
        static constexpr size_t KEY = 1;          ///< [B, Hkv, S, Dh], dense history
        static constexpr size_t VALUE = 2;        ///< [B, Hkv, S, Dv], dense history
        static constexpr size_t SEL_INDICES = 3;  ///< [B, h_sel, ceil(L/q_block_size), K_max]
        static constexpr size_t SEL_COUNT = 4;    ///< [B, h_sel, ceil(L/q_block_size)]
        static constexpr size_t ATTN_MASK = 5;    ///< optional, broadcastable to [B,1,L,S]
        static constexpr size_t SCALE = 6;        ///< optional scalar tensor

        static constexpr size_t COUNT_MIN = 5;
        static constexpr size_t COUNT_MAX = 7;
    };

    struct Config {
        size_t num_heads = 0;
        size_t num_kv_heads = 0;
        size_t h_sel = 1;           ///< 1, num_kv_heads, or num_heads
        size_t k_head_size = 0;
        size_t v_head_size = 0;
        size_t sel_block_size = 0;  ///< matches producer's compress_ratio
        size_t q_block_size = 1;    ///< v1 fixed at 1
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

    bool has_attn_mask() const {
        return get_input_size() >= InputIdx::ATTN_MASK + 1;
    }
    bool has_scale_input() const {
        return get_input_size() >= InputIdx::SCALE + 1;
    }

    static bool validate_config(const Config& config, std::string& reason);

private:
    Config m_config{};
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::op::internal
