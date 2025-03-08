// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/fully_connected.hpp"

#include <memory>

#include "matmul_shape_inference.hpp"

namespace ov {
namespace op {
namespace internal {

FullyConnected::FullyConnected(const ov::Output<Node>& A,
                               const ov::Output<Node>& B,
                               const ov::Output<Node>& bias,
                               const ov::element::Type output_type)
    : Op({A, B, bias}),
      m_output_type(output_type) {
    validate_and_infer_types();
}

FullyConnected::FullyConnected(const ov::Output<Node>& A,
                               const ov::Output<Node>& B,
                               const ov::element::Type output_type)
    : FullyConnected(A, B, std::make_shared<v0::Constant>(element::dynamic, Shape{0}), output_type) {}

bool FullyConnected::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

std::shared_ptr<ov::Node> FullyConnected::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<FullyConnected>(new_args.at(0), new_args.at(1), new_args.at(2), m_output_type);
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
    op.set_transpose_b(true);

    auto out_shapes =
        ov::op::v0::shape_infer(&op,
                                std::vector<ov::PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)});
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
