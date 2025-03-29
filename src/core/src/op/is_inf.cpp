// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_inf.hpp"

#include "itt.hpp"

namespace ov {
op::v10::IsInf::IsInf(const Output<Node>& data) : op::Op{{data}} {
    constructor_validate_and_infer_types();
}

op::v10::IsInf::IsInf(const Output<Node>& data, const Attributes& attributes)
    : op::Op{{data}},
      m_attributes{attributes} {
    constructor_validate_and_infer_types();
}

bool op::v10::IsInf::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v10_IsInf_visit_attributes);
    visitor.on_attribute("detect_negative", m_attributes.detect_negative);
    visitor.on_attribute("detect_positive", m_attributes.detect_positive);
    return true;
}

void op::v10::IsInf::validate_and_infer_types() {
    OV_OP_SCOPE(v10_IsInf_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_dynamic() || get_input_element_type(0).is_real(),
                          "The element type of the input tensor must be a floating point number.");
    set_output_type(0, element::boolean, get_input_partial_shape(0));
}

std::shared_ptr<Node> op::v10::IsInf::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v10_IsInf_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v10::IsInf>(new_args.at(0), this->get_attributes());
}
}  // namespace ov
