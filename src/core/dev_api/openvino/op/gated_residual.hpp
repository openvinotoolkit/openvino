// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>
#include <optional>
#include <ostream>

#include "openvino/core/attribute_adapter.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/op.hpp"

namespace ov::op::internal {

/// \brief Gated Residual (hyper connection) operation. The residual is [B,T,C*D], i.e. [B,T,C,D] flattened.
///
/// READ / FINAL_MIX (branch merge):
///   Rbar_c = rms_norm(R_c) * gain_c           (per-branch RMSNorm over D)
///   G      = sigmoid(W_up * SiLU(W_down * Rbar / C))
///   mixed  = (1/C) * sum_c G_c (*) Rbar_c
///   S      = 2 * sigmoid(W_inject * Rbar / C) (optional, only with produce_write_gate)
/// WRITE (branch update): R'_c = R_c + S_c * Y
///
/// The 1/C factors are model semantics. The norm gain is used as it is: a `1 + w` is folded into it by the exporter.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GatedResidual : public ov::op::Op {
public:
    OPENVINO_OP("GatedResidual");

    enum class Mode {
        READ = 0,       ///< branch merge, optionally also produces the write gate
        WRITE = 1,      ///< branch update using a precomputed write gate
        FINAL_MIX = 2,  ///< model-tail mixer: READ without write-gate generation
    };

    /// @brief Input form of a READ / FINAL_MIX node; the write-gate inputs come last, so their positions depend on it:
    ///   floating-point weights:        residual, norm, w_down, w_up [, w_inject]
    ///   compressed, no zero points:    residual, norm, w_down, w_up, scale_down, scale_up [, w_inject, scale_inject]
    ///   compressed with zero points:   ... scale_up, zp_down, zp_up [, w_inject, scale_inject, zp_inject]
    /// Compressed weights are i4/u4/i8/u8; scale and zero point are [N, G] (G groups along K) or one element. Zero
    /// points are all-or-nothing.
    struct ReadLayout {
        bool compressed = false;
        bool has_zp = false;
        bool write_gate = false;
        size_t input_count = 4;
        size_t residual = 0;      ///< [B,T,C*D]
        size_t norm_weight = 1;   ///< [C*D]
        size_t w_down = 2;        ///< [L, C*D]
        size_t w_up = 3;          ///< [C*D, L]
        size_t scale_down = 4;    ///< valid when compressed: floating-point (f16 / bf16 / f32) [L, G_d]
        size_t scale_up = 5;      ///< valid when compressed: floating-point [C*D, G_u]
        size_t zp_down = 6;       ///< valid when has_zp: weight type [L, G_d]
        size_t zp_up = 7;         ///< valid when has_zp: weight type [C*D, G_u]
        size_t w_inject = 0;      ///< valid when write_gate
        size_t scale_inject = 0;  ///< valid when write_gate and compressed
        size_t zp_inject = 0;     ///< valid when write_gate, compressed and has_zp
    };

    /// The input form selected by the input count; empty when no form has that many inputs.
    std::optional<ReadLayout> get_read_layout() const;

    /// \brief Attributes of the operation.
    struct Config {
        Mode mode{Mode::READ};
        size_t hc_count{4};              ///< C: number of residual branches
        size_t hidden_size{0};           ///< D: per-branch hidden size
        size_t lowrank{0};               ///< L: low rank of the gate (READ / FINAL_MIX only)
        float epsilon{1e-6F};            ///< RMSNorm epsilon
        bool produce_write_gate{false};  ///< READ also outputs the write gate S
    };

    GatedResidual() = default;

    /// \brief Constructs a GatedResidual operation.
    /// \param args The input tensors; their layout depends on the mode and weight form, see ReadLayout and the
    ///        WRITE mode inputs: residual [B,T,C*D], sublayer output Y [B,T,D], write gate [B,T,C].
    /// \param config Attributes of the operation
    GatedResidual(const ov::OutputVector& args, const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const;

private:
    Config m_config;
};

}  // namespace ov::op::internal

namespace ov {
std::ostream& operator<<(std::ostream& s, const ov::op::internal::GatedResidual::Mode& mode);

template <>
class AttributeAdapter<ov::op::internal::GatedResidual::Mode>
    : public EnumAttributeAdapterBase<ov::op::internal::GatedResidual::Mode> {
public:
    AttributeAdapter(ov::op::internal::GatedResidual::Mode& value)
        : EnumAttributeAdapterBase<ov::op::internal::GatedResidual::Mode>(value) {}

    OPENVINO_RTTI("AttributeAdapter<ov::op::internal::GatedResidual::Mode>");
    ~AttributeAdapter() override = default;
};
}  // namespace ov
