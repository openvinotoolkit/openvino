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
                               const bool transpose_b)
    : Op({A, B, bias}), m_output_type(output_type), m_transpose_b(transpose_b) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> FullyConnected::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<FullyConnected>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_type, m_transpose_b);
}

void FullyConnected::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size >= 3,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected at least 3.");

    ov::op::v0::MatMul op;
    op.set_transpose_a(false);
    op.set_transpose_b(m_transpose_b);

    // Group-major u2 compressed weights are a genuine 3-D [KG, N, GS] layout (GS = 64, the CM u2 kernel's
    // quantization group size). Collapse to the logical [N, K] = [N, KG*GS] matrix for the MatMul-based
    // shape inference below -- it would otherwise read KG as a batch dim and reject the K mismatch. The CM
    // kernel reads the untouched 3-D memory in group-major order (WLAYOUT 0).
    auto weights_pshape = get_input_partial_shape(1);
    if (get_input_element_type(1) == ov::element::u2 && weights_pshape.rank().is_static() && weights_pshape.size() == 3 &&
        weights_pshape[0].is_static() && weights_pshape[1].is_static() && weights_pshape[2].is_static() &&
        weights_pshape[2].get_length() == 64) {
        const auto N = weights_pshape[1].get_length();
        const auto K = weights_pshape[0].get_length() * weights_pshape[2].get_length();
        weights_pshape = ov::PartialShape{N, K};
    }

    auto out_shapes = ov::op::v0::shape_infer(&op, std::vector<ov::PartialShape>{get_input_partial_shape(0), weights_pshape});

    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool FullyConnected::visit_attributes(ov::AttributeVisitor &visitor) {
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("transpose_b", m_transpose_b);
    return true;
}

}  // namespace ov::intel_gpu::op
