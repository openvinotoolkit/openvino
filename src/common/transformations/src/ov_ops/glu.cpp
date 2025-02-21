// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_ops/glu.hpp"

#include "glu_shape_inference.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace internal {

GLU::GLU(const Output<Node>& data,
         int64_t axis,
         int64_t split_lengths,
         const GluType glu_type,
         const size_t split_to_glu_idx,
         const ov::element::Type output_type)
    : Op({data}),
      m_axis(axis),
      m_split_lengths(split_lengths),
      m_glu_type(glu_type),
      m_split_to_glu_idx(split_to_glu_idx),
      m_output_type(output_type) {
    validate_and_infer_types();
}

bool GLU::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("split_lengths", m_split_lengths);
    visitor.on_attribute("glu_type", m_glu_type);
    visitor.on_attribute("split_to_glu_idx", m_split_to_glu_idx);
    visitor.on_attribute("output_type", m_output_type);
    return true;
}

void GLU::validate_and_infer_types() {
    auto output_type = m_output_type == ov::element::dynamic ? get_input_element_type(0) : m_output_type;

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, output_type, output_shapes[0]);
}

std::shared_ptr<Node> GLU::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<GLU>(new_args.at(0),
                                 m_axis,
                                 m_split_lengths,
                                 m_glu_type,
                                 m_split_to_glu_idx,
                                 m_output_type);
}
}  // namespace internal
}  // namespace op

template <>
OPENVINO_API EnumNames<op::internal::GLU::GluType>& EnumNames<op::internal::GLU::GluType>::get() {
    static auto enum_names =
        EnumNames<op::internal::GLU::GluType>("op::internal::GLU::GluType",
                                              {{"Swish", op::internal::GLU::GluType::Swish},
                                               {"Gelu", op::internal::GLU::GluType::Gelu},
                                               {"Gelu_Tanh", op::internal::GLU::GluType::Gelu_Tanh}});
    return enum_names;
}

}  // namespace ov
