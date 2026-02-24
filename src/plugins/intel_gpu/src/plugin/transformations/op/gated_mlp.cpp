// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/gated_mlp.hpp"

#include "matmul_shape_inference.hpp"

namespace ov::intel_gpu::op {

GatedMLP::GatedMLP(const ov::Output<Node>& src,
                   const ov::Output<Node>& w_gate,
                   const ov::Output<Node>& w_up,
                   const ov::Output<Node>& w_down,
                   ov::op::internal::GLU::GluType activation,
                   const ov::element::Type output_type)
    : Op({src, w_gate, w_up, w_down}),
      m_activation(activation),
      m_output_type(output_type) {
    validate_and_infer_types();
}

bool GatedMLP::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("activation", m_activation);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void GatedMLP::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 4,
                          "GatedMLP expects 4 inputs (src, w_gate, w_up, w_down), got ",
                          get_input_size());

    const auto& src_ps = get_input_partial_shape(0);
    const auto& w_gate_ps = get_input_partial_shape(1);
    const auto& w_up_ps = get_input_partial_shape(2);
    const auto& w_down_ps = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          src_ps.rank().is_dynamic() || src_ps.rank().get_length() >= 2,
                          "GatedMLP supports src rank >= 2.");
    NODE_VALIDATION_CHECK(this, w_gate_ps.rank().compatible(2), "GatedMLP supports rank-2 w_gate only.");
    NODE_VALIDATION_CHECK(this, w_up_ps.rank().compatible(2), "GatedMLP supports rank-2 w_up only.");
    NODE_VALIDATION_CHECK(this, w_down_ps.rank().compatible(2), "GatedMLP supports rank-2 w_down only.");

    ov::op::v0::MatMul matmul;
    matmul.set_transpose_a(false);
    matmul.set_transpose_b(false);

    auto up_shapes = ov::op::v0::shape_infer(&matmul, std::vector<ov::PartialShape>{src_ps, w_up_ps});
    auto gate_shapes = ov::op::v0::shape_infer(&matmul, std::vector<ov::PartialShape>{src_ps, w_gate_ps});

    NODE_VALIDATION_CHECK(this,
                          up_shapes[0].compatible(gate_shapes[0]),
                          "GatedMLP requires gate/up projection output shapes to match.");

    auto out_shapes = ov::op::v0::shape_infer(&matmul, std::vector<ov::PartialShape>{up_shapes[0], w_down_ps});
    auto out_et = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, out_et, out_shapes[0]);
}

std::shared_ptr<Node> GatedMLP::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GatedMLP>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_activation, m_output_type);
}

}  // namespace ov::intel_gpu::op
