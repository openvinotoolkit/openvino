// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/gated_residual.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal Gated Residual (Hyper Connection) operation.
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
/// This class reuses ov::op::internal::GatedResidual's Mode / InputIdx / shape-inference /
/// attribute visiting logic wholesale (identical semantics); the only GPU-specific bit is
/// the "gpu_opset" RTTI tag consumed by cldnn::gated_residual's primitive lowering (see
/// intel_gpu/src/plugin/ops/gated_residual.cpp, intel_gpu/src/graph/gated_residual.cpp).
class GatedResidual : public ov::op::internal::GatedResidual {
public:
    OPENVINO_OP("GatedResidual", "gpu_opset", ov::op::internal::GatedResidual);

    using Base = ov::op::internal::GatedResidual;
    using Mode = Base::Mode;
    using ReadInputIdx = Base::ReadInputIdx;
    using WriteInputIdx = Base::WriteInputIdx;

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
                  const ov::element::Type output_type = ov::element::dynamic)
        : Base(args, mode, hc_count, hidden_size, lowrank, epsilon, norm_gain_plus_one, produce_write_gate, output_type) {}

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
