// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/itt.hpp"

#include "snippets/op/fill.hpp"


namespace ov {
namespace snippets {
namespace op {

Fill::Fill(const Output<Node>& x, const size_t offset, const uint32_t fill_value)
    : Op({x}), m_offset(offset), m_fill_value(fill_value) {
    constructor_validate_and_infer_types();
}

bool Fill::visit_attributes(AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(Fill_visit_attributes);
    visitor.on_attribute("offset", m_offset);
    visitor.on_attribute("fill_value", m_fill_value);
    return true;
}

std::shared_ptr<Node> Fill::clone_with_new_inputs(const OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(Fill_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Fill>(new_args.at(0), m_offset, m_fill_value);
}

void Fill::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(Fill_validate_and_infer_types);
    const auto in_type = get_input_element_type(0);
    OPENVINO_ASSERT(in_type.size() == 4, "Fill operation supports only element types with 4 byte size but got:" + std::to_string(in_type.size()));
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

} // namespace op
} // namespace snippets
} // namespace ov

