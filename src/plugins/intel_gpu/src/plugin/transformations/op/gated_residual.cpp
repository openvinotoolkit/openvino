// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gated_residual.hpp"

#include "openvino/core/validation_util.hpp"

namespace ov::intel_gpu::op {

GatedResidual::GatedResidual(const ov::OutputVector& args,
                             Mode mode,
                             size_t hc_count,
                             size_t hidden_size,
                             size_t lowrank,
                             float epsilon,
                             bool norm_gain_plus_one,
                             bool produce_write_gate,
                             const ov::element::Type output_type)
    : Op(args),
      m_mode(mode),
      m_hc_count(hc_count),
      m_hidden_size(hidden_size),
      m_lowrank(lowrank),
      m_epsilon(epsilon),
      m_norm_gain_plus_one(norm_gain_plus_one),
      m_produce_write_gate(produce_write_gate),
      m_output_type(output_type) {
    validate_and_infer_types();
}

bool GatedResidual::visit_attributes(ov::AttributeVisitor& visitor) {
    auto mode = static_cast<int64_t>(m_mode);
    visitor.on_attribute("mode", mode);
    m_mode = static_cast<Mode>(mode);
    visitor.on_attribute("hc_count", m_hc_count);
    visitor.on_attribute("hidden_size", m_hidden_size);
    visitor.on_attribute("lowrank", m_lowrank);
    visitor.on_attribute("epsilon", m_epsilon);
    visitor.on_attribute("norm_gain_plus_one", m_norm_gain_plus_one);
    visitor.on_attribute("produce_write_gate", m_produce_write_gate);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void GatedResidual::validate_and_infer_types() {
    const auto input_size = get_input_size();
    const auto out_et = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;

    NODE_VALIDATION_CHECK(this, m_hc_count > 0, "GatedResidual requires a positive branch count C.");
    NODE_VALIDATION_CHECK(this, m_hidden_size > 0, "GatedResidual requires a positive hidden size D.");

    const auto& residual_ps = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this,
                          residual_ps.rank().is_dynamic() || residual_ps.rank().get_length() >= 2,
                          "GatedResidual requires the residual input rank to be at least 2.");

    if (m_mode == Mode::WRITE) {
        NODE_VALIDATION_CHECK(this, input_size == 3, "GatedResidual (WRITE) expects 3 inputs, got ", input_size);
        // R' has exactly the residual layout.
        set_output_size(1);
        set_output_type(0, out_et, residual_ps);
        return;
    }

    // READ / FINAL_MIX
    NODE_VALIDATION_CHECK(this,
                          (m_produce_write_gate && input_size == 5) || (!m_produce_write_gate && input_size == 4),
                          "GatedResidual (READ/FINAL_MIX) expects 4 inputs (5 with W_inject), got ",
                          input_size);
    NODE_VALIDATION_CHECK(this, m_mode != Mode::FINAL_MIX || !m_produce_write_gate, "GatedResidual (FINAL_MIX) must not produce a write gate.");
    NODE_VALIDATION_CHECK(this, m_lowrank > 0, "GatedResidual (READ/FINAL_MIX) requires a positive low rank L.");

    auto mixed_ps = residual_ps;
    if (mixed_ps.rank().is_static()) {
        mixed_ps[mixed_ps.rank().get_length() - 1] = ov::Dimension(static_cast<int64_t>(m_hidden_size));
    }
    set_output_size(m_produce_write_gate ? 2 : 1);
    set_output_type(0, out_et, mixed_ps);

    if (m_produce_write_gate) {
        auto gate_ps = residual_ps;
        if (gate_ps.rank().is_static()) {
            gate_ps[gate_ps.rank().get_length() - 1] = ov::Dimension(static_cast<int64_t>(m_hc_count));
        }
        set_output_type(1, out_et, gate_ps);
    }
}

std::shared_ptr<Node> GatedResidual::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GatedResidual>(new_args,
                                           m_mode,
                                           m_hc_count,
                                           m_hidden_size,
                                           m_lowrank,
                                           m_epsilon,
                                           m_norm_gain_plus_one,
                                           m_produce_write_gate,
                                           m_output_type);
}

}  // namespace ov::intel_gpu::op
