// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/inverse.hpp"

#include <cstring>

#include "inverse_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/reference/inverse.hpp"

// ------------------------------ v14 ------------------------------

namespace ov {

op::v14::Inverse::Inverse(const Output<Node>& data, const bool adjoint) : Op({data}), m_adjoint(adjoint) {
    constructor_validate_and_infer_types();
}

bool op::v14::Inverse::Inverse::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v14_Inverse_visit_attributes);
    visitor.on_attribute("adjoint", m_adjoint);
    return true;
}

void op::v14::Inverse::Inverse::validate_and_infer_types() {
    OV_OP_SCOPE(v14_Inverse_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(0).is_real() || get_input_element_type(0).is_dynamic(),
                          "Expected floating point type as element type for the 'data' input.");

    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

std::shared_ptr<Node> op::v14::Inverse::Inverse::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v14_Inverse_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<op::v14::Inverse>(new_args.at(0), m_adjoint);
}

bool op::v14::Inverse::get_adjoint() const {
    return m_adjoint;
}

void op::v14::Inverse::set_adjoint(const bool adjoint) {
    m_adjoint = adjoint;
}
}  // namespace ov
