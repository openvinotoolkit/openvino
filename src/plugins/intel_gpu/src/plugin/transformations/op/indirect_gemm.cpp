// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/indirect_gemm.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov::intel_gpu::op {

IndirectGemm::IndirectGemm(const ov::Output<Node>& A,
                           const ov::Output<Node>& B,
                           const ov::Output<Node>& I,
                           bool indirect_a,
                           bool indirect_b,
                           int64_t indirect_axis,
                           const std::vector<int64_t>& order_a,
                           const std::vector<int64_t>& order_b,
                           const std::vector<int64_t>& order_c,
                           const ov::element::Type output_type)
    : ov::intel_gpu::op::Gemm(A, B, order_a, order_b, order_c, output_type)
    , m_indirect_a(indirect_a)
    , m_indirect_b(indirect_b)
    , m_indirect_axis(indirect_axis) {
    set_argument(2, I);
    OPENVINO_ASSERT((indirect_a && indirect_b) == false, "[GPU] Gemm supports indirect addressing for one input only");
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> IndirectGemm::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    return std::make_shared<IndirectGemm>(new_args.at(0),
                                          new_args.at(1),
                                          new_args.at(2),
                                          m_indirect_a,
                                          m_indirect_b,
                                          m_indirect_axis,
                                          m_order_a,
                                          m_order_b,
                                          m_order_c,
                                          m_output_type);
}

void IndirectGemm::validate_and_infer_types() {
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 3,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected 3.");

    auto out_shapes = shape_infer(this,
                                  std::vector<ov::PartialShape>{get_input_partial_shape(0), get_input_partial_shape(1)},
                                  m_order_a,
                                  m_order_b,
                                  m_order_c);

    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, out_shapes[0]);
}

bool IndirectGemm::visit_attributes(ov::AttributeVisitor &visitor) {
    Gemm::visit_attributes(visitor);
    visitor.on_attribute("indirect_a", m_indirect_a);
    visitor.on_attribute("indirect_b", m_indirect_b);
    visitor.on_attribute("indirect_axis", m_indirect_axis);
    return true;
}

}  // namespace ov::intel_gpu::op
