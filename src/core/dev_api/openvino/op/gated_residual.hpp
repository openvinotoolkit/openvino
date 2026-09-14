// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \note GatedResidual op class is under development and subject to change.
///
/// \brief Gated Residual (Hyper Connection) operation.
///
/// The op models the two halves of the gated-residual mechanism used by
/// Qwen4-Exp / Qwen3.8-Flash-Next style models, where the residual stream is widened
/// from [B,T,D] to [B,T,C,D] (physically stored as [B,T,C*D]).
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

    /// @brief Input indices for Mode::READ / Mode::FINAL_MIX
    struct ReadInputIdx {
        static constexpr size_t RESIDUAL = 0;     ///< [B,T,C*D]
        static constexpr size_t NORM_WEIGHT = 1;  ///< [C*D]
        static constexpr size_t W_DOWN = 2;       ///< [L, C*D]
        static constexpr size_t W_UP = 3;         ///< [C*D, L]
        static constexpr size_t W_INJECT = 4;     ///< [C, C*D], only when produce_write_gate
    };

    /// @brief Input indices for Mode::WRITE
    struct WriteInputIdx {
        static constexpr size_t RESIDUAL = 0;    ///< [B,T,C*D]
        static constexpr size_t SUBLAYER_Y = 1;  ///< [B,T,D]
        static constexpr size_t WRITE_GATE = 2;  ///< [B,T,C]
    };

    GatedResidual() = default;

    /// @brief READ / FINAL_MIX constructor.
    GatedResidual(const ov::OutputVector& args,
                  Mode mode,
                  size_t hc_count,
                  size_t hidden_size,
                  size_t lowrank,
                  float epsilon,
                  bool norm_gain_plus_one,
                  bool produce_write_gate,
                  const ov::element::Type output_type = ov::element::dynamic);

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
    ov::element::Type get_output_type() const {
        return m_output_type;
    }

private:
    Mode m_mode = Mode::READ;
    size_t m_hc_count = 4;
    size_t m_hidden_size = 0;
    size_t m_lowrank = 0;
    float m_epsilon = 1e-6f;
    bool m_norm_gain_plus_one = true;
    bool m_produce_write_gate = false;
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::op::internal
