// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/parameter.hpp"

#include <sstream>

#include "itt.hpp"
#include "layout_utils.hpp"
#include "openvino/core/attribute_visitor.hpp"

namespace ov {

op::v0::Parameter::Parameter(const element::Type& element_type, const ov::PartialShape& pshape)
    : m_partial_shape(pshape),
      m_element_type(element_type),
      m_is_relevant_to_shapes(false) {
    constructor_validate_and_infer_types();
}

bool op::v0::Parameter::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_Parameter_visit_attributes);
    visitor.on_attribute("shape", m_partial_shape);
    visitor.on_attribute("element_type", m_element_type);
    return true;
}

void op::v0::Parameter::validate_and_infer_types() {
    OV_OP_SCOPE(v0_Parameter_validate_and_infer_types);
    Op::validate_and_infer_types();
    set_output_type(0, m_element_type, m_partial_shape);
}

std::shared_ptr<Node> op::v0::Parameter::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_Parameter_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Parameter>(m_element_type, m_partial_shape);
}

bool op::v0::Parameter::is_relevant_to_shapes() const {
    return m_is_relevant_to_shapes;
}

void op::v0::Parameter::set_is_relevant_to_shapes(bool is_relevant) {
    m_is_relevant_to_shapes = is_relevant;
}

ov::Layout op::v0::Parameter::get_layout() const {
    return ov::layout::get_layout(output(0));
}

void op::v0::Parameter::set_layout(const ov::Layout& layout) {
    ov::layout::set_layout(output(0), layout);
}

void op::v0::Parameter::set_partial_shape(const PartialShape& partial_shape) {
    OPENVINO_ASSERT(ov::layout::utils::is_compatible(get_layout(), partial_shape),
                    "Can't set partial shape ",
                    partial_shape,
                    " for Parameter ",
                    *this,
                    " with layout ",
                    get_layout().to_string(),
                    ". Layout is not compatible with shape");
    m_partial_shape = partial_shape;
}

AttributeAdapter<ParameterVector>::AttributeAdapter(ParameterVector& ref) : m_ref(ref) {}

AttributeAdapter<ParameterVector>::~AttributeAdapter() = default;

bool AttributeAdapter<ParameterVector>::visit_attributes(AttributeVisitor& visitor) {
    size_t size = m_ref.size();
    visitor.on_attribute("size", size);
    if (size != m_ref.size()) {
        m_ref.resize(size);
    }
    std::ostringstream index;
    for (size_t i = 0; i < size; i++) {
        index.str("");
        index << i;
        std::string id;
        if (m_ref[i]) {
            id = visitor.get_registered_node_id(m_ref[i]);
        }
        visitor.on_attribute(index.str(), id);
        if (!m_ref[i]) {
            m_ref[i] = ov::as_type_ptr<op::v0::Parameter>(visitor.get_registered_node(std::move(id)));
        }
    }
    return true;
}
}  // namespace ov
