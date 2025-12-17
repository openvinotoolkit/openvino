// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace op {
namespace v0 {
/// \brief Cyberspore TSSN operation.
///
/// Implements the Mamba-inspired SSM equation h_t = A_bar h_{t-1} + B_bar x_t
/// utilizing balanced ternary logic (-1, 0, +1).
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API CybersporeTSSN : public Op {
public:
    OPENVINO_OP("CybersporeTSSN", "opset1", Op);

    CybersporeTSSN() = default;

    /// \brief Constructs a CybersporeTSSN operation.
    ///
    /// \param input_events  Input tensor (Ternary).
    /// \param state_matrix  State tensor (Ternary).
    /// \param selective_params Selective parameters (Float/BF16).
    /// \param homeostatic_setpoint Homeostatic Setpoint (nu).
    /// \param decay_rate Decay Rate.
    CybersporeTSSN(const Output<Node>& input_events,
                   const Output<Node>& state_matrix,
                   const Output<Node>& selective_params,
                   float homeostatic_setpoint,
                   float decay_rate);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    float get_homeostatic_setpoint() const { return m_homeostatic_setpoint; }
    float get_decay_rate() const { return m_decay_rate; }
    void set_homeostatic_setpoint(float homeostatic_setpoint) { m_homeostatic_setpoint = homeostatic_setpoint; }
    void set_decay_rate(float decay_rate) { m_decay_rate = decay_rate; }

    bool visit_attributes(AttributeVisitor& visitor) override;

private:
    float m_homeostatic_setpoint;
    float m_decay_rate;
};
}  // namespace v0
}  // namespace op
}  // namespace ov
