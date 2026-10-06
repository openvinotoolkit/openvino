// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/rms.hpp"

namespace ov {
namespace op {
namespace internal {

RMS::RMS(const Output<Node>& data,
         const Output<Node>& gamma,
         double epsilon,
         const ov::element::Type output_type,
         int64_t axis)
    : Op({data, gamma}),
      m_epsilon(epsilon),
      m_output_type(output_type),
      m_elementwise_affine(true),
      m_axis(axis) {
    validate_and_infer_types();
}

RMS::RMS(const Output<Node>& data, double epsilon, const ov::element::Type output_type, int64_t axis)
    : Op({data}),
      m_epsilon(epsilon),
      m_output_type(output_type),
      m_elementwise_affine(false),
      m_axis(axis) {
    validate_and_infer_types();
}

bool RMS::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("epsilon", m_epsilon);
    visitor.on_attribute("output_type", m_output_type);
    visitor.on_attribute("elementwise_affine", m_elementwise_affine);
    visitor.on_attribute("axis", m_axis);
    return true;
}

void RMS::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;
    set_output_type(0, output_type, get_input_partial_shape(0));
}

std::shared_ptr<Node> RMS::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    if (new_args.size() == 1) {
        return std::make_shared<RMS>(new_args.at(0), m_epsilon, m_output_type, m_axis);
    }
    return std::make_shared<RMS>(new_args.at(0), new_args.at(1), m_epsilon, m_output_type, m_axis);
}

}  // namespace internal
}  // namespace op
}  // namespace ov
