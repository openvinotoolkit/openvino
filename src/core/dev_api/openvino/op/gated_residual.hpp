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

/// \brief Gated Residual (Hyper Connection) operation.
///
/// The op models the two halves of the gated-residual (hyper connection) mechanism, where the residual stream is
/// widened from [B,T,D] to [B,T,C,D] (physically stored as [B,T,C*D]).
///
/// READ / FINAL_MIX (branch merge, executed *before* a sub-layer):
///   Rbar_c = rms_norm(R_c) * gain_c                (branch-wise grouped RMSNorm, group width D)
///   G      = sigmoid(W_up * SiLU(W_down * Rbar / C))
///   mixed  = (1/C) * sum_c G_c (*) Rbar_c
///   S      = 2 * sigmoid(W_inject * Rbar / C)      (optional, only when produce_write_gate)
///
/// WRITE (branch update, executed *after* a sub-layer):
///   R'_c = R_c + S_c * Y
///
/// Notes on semantics that must not be "optimized away":
///  - the normalization is per-branch over D channels, *not* a single RMSNorm over C*D;
///  - all three 1/C factors are part of the model semantics;
///  - the read gate is per-(branch, channel), the write gate is a per-branch scalar in (0,2);
///  - `gain = 1 + w` when `norm_gain_plus_one` is set (checkpoint stores w initialized to 0).
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

        // Compressed weights (W_DOWN / W_UP are i4, u4, i8 or u8). scale/zp are [G, N] (N contiguous), unlike FC's [N,
        // G]: one sub-group block read gives every lane the scale of its own output channel.
        static constexpr size_t SCALE_DOWN = 4;  ///< floating-point (f16 / bf16 / f32) [G_d, L]
        static constexpr size_t SCALE_UP = 5;    ///< floating-point (f16 / bf16 / f32) [G_u, C*D]
        static constexpr size_t ZP_DOWN =
            6;  ///< same type as the weights (4-bit packed) [G_d, L], only with zero points
        static constexpr size_t ZP_UP =
            7;  ///< same type as the weights (4-bit packed) [G_u, C*D], only with zero points
    };

    /// @brief Input form of a READ / FINAL_MIX node. The write-gate inputs are always the last inputs, so their indices
    /// depend on the form:
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

    std::optional<ReadLayout> get_read_layout() const {
        return read_layout(get_input_size(), m_produce_write_gate);
    }

    static bool is_compressed_weight_type(const ov::element::Type& t) {
        return t == ov::element::i4 || t == ov::element::u4 || t == ov::element::i8 || t == ov::element::u8;
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
    Mode m_mode = Mode::READ;
    size_t m_hc_count = 4;
    size_t m_hidden_size = 0;
    size_t m_lowrank = 0;
    float m_epsilon = 1e-6f;
    bool m_norm_gain_plus_one = true;
    bool m_produce_write_gate = false;
};

}  // namespace ov::op::internal
