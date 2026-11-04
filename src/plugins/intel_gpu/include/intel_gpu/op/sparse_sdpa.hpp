// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>

#include "intel_gpu/op/sparse_kv_quant.hpp"
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

    /// @brief Fixed input order: the six core inputs, then the parameters of a quantized cache, then the optional SCALE and
    /// SINKS (see Config::scale_input_idx() / sinks_input_idx() for their positions).
    struct InputIdx {
        static constexpr size_t QUERY = 0;                   ///< [B, Hq, L, Dh]
        static constexpr size_t KEY = 1;                     ///< [B, Hkv, S, Dh], dense history
        static constexpr size_t VALUE = 2;                   ///< [B, Hkv, S, Dv], dense history
        static constexpr size_t SELECTED_BLOCK_INDICES = 3;  ///< [B, num_selection_heads, ceil(L/query_block_size), K_max]
        static constexpr size_t SELECTED_BLOCK_COUNT = 4;    ///< [B, num_selection_heads, ceil(L/query_block_size)]
        static constexpr size_t KEY_LENGTH = 5;              ///< [B] i32, valid keys per batch row
        static constexpr size_t COUNT_CORE = 6;
        /// Only with a quantized cache (`Config::kv_compressed()`): KEY/VALUE are i8 and the parameters follow in the
        /// KVCacheCompressed planar layout (see sparse_kv_quant.hpp). The zero points are absent when symmetric.
        static constexpr size_t KEY_SCALE = 6;    ///< [B, Hkv, S, 1] f16
        static constexpr size_t VALUE_SCALE = 7;  ///< [B, Hkv, S, 1] f16
        static constexpr size_t KEY_ZP = 8;       ///< same shape as KEY_SCALE, i8 (8-bit BY_TOKEN) or f16
        static constexpr size_t VALUE_ZP = 9;     ///< same shape as VALUE_SCALE
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
        KVQuant key_quant{};    ///< quantization of the key history (the K and V histories share bits and symmetry)
        KVQuant value_quant{};  ///< quantization of the value history

        bool kv_compressed() const {
            return key_quant.enabled() || value_quant.enabled();
        }
        /// Inputs between the core ones and the optional SCALE: scales, plus zero points unless symmetric.
        size_t param_input_count() const {
            return !kv_compressed() ? 0 : key_quant.symmetric ? 2 : 4;
        }
        /// Optional scalar or 1-element tensor.
        size_t scale_input_idx() const {
            return InputIdx::COUNT_CORE + param_input_count();
        }
        /// Optional [num_heads] input after SCALE; not supported with a quantized cache.
        size_t sinks_input_idx() const {
            return scale_input_idx() + 1;
        }
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
        return get_input_size() > m_config.scale_input_idx();
    }
    bool has_sinks() const {
        return !m_config.kv_compressed() && get_input_size() > m_config.sinks_input_idx();
    }

    static bool validate_config(const Config& config, std::string& reason);

private:
    Config m_config{};
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::intel_gpu::op
