// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/fully_connected.hpp"
#include "matmul_shape_inference.hpp"

namespace ov::intel_gpu::op {

FullyConnected::FullyConnected(const ov::Output<Node>& A,
                               const ov::Output<Node>& B,
                               const ov::Output<Node>& bias,
                               const ov::element::Type output_type,
                               const bool transpose_b,
                               const std::optional<FcWeightsLayout>& weights_layout)
    : Op({A, B, bias}),
      m_output_type(output_type),
      m_transpose_b(transpose_b),
      m_weights_layout(weights_layout.value_or(fc_default_weights_layout(transpose_b))) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> FullyConnected::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<FullyConnected>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_type, m_transpose_b, m_weights_layout);
}

void FullyConnected::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size >= 3,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected at least 3.");

    NODE_VALIDATION_CHECK(this,
        fc_weights_layout_is_consistent(m_weights_layout, m_transpose_b),
        "weights_layout ",
        to_string(m_weights_layout),
        " is inconsistent with transpose_b = ",
        m_transpose_b);

    auto weights_pshape = get_input_partial_shape(1);
    if (m_weights_layout == FcWeightsLayout::GNK) {
        const auto dims = fc_weight_dims(weights_pshape, m_weights_layout);
        weights_pshape = ov::PartialShape{dims.N, dims.K};
    }

    ov::op::v0::MatMul op;
    op.set_transpose_a(false);
    op.set_transpose_b(m_transpose_b);

    auto out_shapes = ov::op::v0::shape_infer(&op, std::vector<ov::PartialShape>{get_input_partial_shape(0), weights_pshape});

    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool FullyConnected::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("transpose_b", m_transpose_b);
    // Falls back to NK / KN from transpose_b if the attribute is absent.
    std::string weights_layout = to_string(m_weights_layout == FcWeightsLayout::GNK ? m_weights_layout : fc_default_weights_layout(m_transpose_b));
    visitor.on_attribute("weights_layout", weights_layout);
    m_weights_layout = fc_weights_layout_from_string(weights_layout);
    return true;
}

}  // namespace ov::intel_gpu::op
