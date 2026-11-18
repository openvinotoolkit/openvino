// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <optional>

#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Gated Residual (hyper connection) operation. The residual is [B,T,C*D], i.e. [B,T,C,D] flattened.
///
/// READ / FINAL_MIX (branch merge):
///   Rbar_c = rms_norm(R_c) * gain_c           (per-branch RMSNorm over D, not over C*D)
///   G      = sigmoid(W_up * SiLU(W_down * Rbar / C))
///   mixed  = (1/C) * sum_c G_c (*) Rbar_c
///   S      = 2 * sigmoid(W_inject * Rbar / C) (optional, only with produce_write_gate)
/// WRITE (branch update): R'_c = R_c + S_c * Y
///
/// The 1/C factors are model semantics. The read gate is per (branch, channel), the write gate per branch.
/// `gain = 1 + w` when `norm_gain_plus_one` is set.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatedResidual : public ov::op::Op {
public:
    OPENVINO_OP("GatedResidual");

    enum class Mode : int32_t {
        READ = 0,       ///< branch merge, optionally also produces the write gate
        WRITE = 1,      ///< branch update using a precomputed write gate
        FINAL_MIX = 2,  ///< model-tail mixer: READ without write-gate generation
    };

    /// @brief Input indices for Mode::READ / Mode::FINAL_MIX that do not depend on the input form
    struct ReadInputIdx {
        static constexpr size_t RESIDUAL = 0;     ///< [B,T,C*D]
        static constexpr size_t NORM_WEIGHT = 1;  ///< [C*D]
        static constexpr size_t W_DOWN = 2;       ///< [L, C*D]
        static constexpr size_t W_UP = 3;         ///< [C*D, L]

        // Compressed weights are i4/u4/i8/u8. scale / zp are [N, G]: N output channels, G groups along K.
        static constexpr size_t SCALE_DOWN = 4;  ///< floating-point (f16 / bf16 / f32) [L, G_d]
        static constexpr size_t SCALE_UP = 5;    ///< floating-point (f16 / bf16 / f32) [C*D, G_u]
        static constexpr size_t ZP_DOWN = 6;     ///< weight type [L, G_d], only with zero points
        static constexpr size_t ZP_UP = 7;       ///< weight type [C*D, G_u], only with zero points
    };

    /// @brief Input form of a READ / FINAL_MIX node; the write-gate inputs come last, so their indices depend on it:
    ///   floating-point weights:        residual, norm, w_down, w_up [, w_inject]
    ///   compressed, no zero points:    residual, norm, w_down, w_up, scale_down, scale_up [, w_inject, scale_inject]
    ///   compressed with zero points:   ... scale_up, zp_down, zp_up [, w_inject, scale_inject, zp_inject]
    /// Zero points are all-or-nothing; without them they are 0.
    struct ReadLayout {
        bool compressed = false;
        bool has_zp = false;
        bool write_gate = false;
        size_t input_count = 4;
        size_t w_inject = 0;      ///< valid when write_gate
        size_t scale_inject = 0;  ///< valid when write_gate and compressed
        size_t zp_inject = 0;     ///< valid when write_gate, compressed and has_zp
    };

    /// The input form selected by the input count; empty when no form has that many inputs.
    std::optional<ReadLayout> get_read_layout() const {
        return read_layout(get_input_size(), m_produce_write_gate);
    }

    /// @brief Input indices for Mode::WRITE
    struct WriteInputIdx {
        static constexpr size_t RESIDUAL = 0;    ///< [B,T,C*D]
        static constexpr size_t SUBLAYER_Y = 1;  ///< [B,T,D]
        static constexpr size_t WRITE_GATE = 2;  ///< [B,T,C]
    };

    GatedResidual() = default;

    GatedResidual(const ov::OutputVector& args,
                  Mode mode,
                  size_t hc_count,
                  size_t hidden_size,
                  size_t lowrank,
                  float epsilon,
                  bool norm_gain_plus_one,
                  bool produce_write_gate);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    Mode get_mode() const {
        return m_mode;
    }
    size_t get_hc_count() const {
        return m_hc_count;
    }
    size_t get_hidden_size() const {
        return m_hidden_size;
    }
    size_t get_lowrank() const {
        return m_lowrank;
    }
    float get_epsilon() const {
        return m_epsilon;
    }
    bool get_norm_gain_plus_one() const {
        return m_norm_gain_plus_one;
    }
    bool get_produce_write_gate() const {
        return m_produce_write_gate;
    }

private:
    static constexpr ReadLayout make_read_layout(bool compressed, bool has_zp, bool write_gate) {
        ReadLayout l;
        l.compressed = compressed;
        l.has_zp = compressed && has_zp;
        l.write_gate = write_gate;
        const size_t base = compressed ? (l.has_zp ? 8 : 6) : 4;
        l.w_inject = base;
        l.scale_inject = base + 1;
        l.zp_inject = base + 2;
        l.input_count = base + (write_gate ? (compressed ? (l.has_zp ? 3 : 2) : 1) : 0);
        return l;
    }

    /// The form selected by the input count; empty when no form has that many inputs.
    static std::optional<ReadLayout> read_layout(size_t input_count, bool write_gate) {
        for (const bool compressed : {false, true}) {
            for (const bool has_zp : {false, true}) {
                const auto l = make_read_layout(compressed, has_zp, write_gate);
                if (l.has_zp == has_zp && l.input_count == input_count) {
                    return l;
                }
            }
        }
        return std::nullopt;
    }

    static bool is_compressed_weight_type(const ov::element::Type& t) {
        return t == ov::element::i4 || t == ov::element::u4 || t == ov::element::i8 || t == ov::element::u8;
    }

    Mode m_mode = Mode::READ;
    size_t m_hc_count = 4;
    size_t m_hidden_size = 0;
    size_t m_lowrank = 0;
    float m_epsilon = 1e-6f;
    bool m_norm_gain_plus_one = true;
    bool m_produce_write_gate = false;
};

}  // namespace ov::op::internal
