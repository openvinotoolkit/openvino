// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/grn.hpp"

#include "itt.hpp"
#include "openvino/core/axis_set.hpp"

namespace ov {

op::v0::GRN::GRN(const Output<Node>& data, float bias) : util::UnaryElementwiseArithmetic(data), m_bias(bias) {
    constructor_validate_and_infer_types();
}

bool op::v0::GRN::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_GRN_visit_attributes);
    visitor.on_attribute("bias", m_bias);
    return true;
}

void op::v0::GRN::validate_and_infer_types() {
    OV_OP_SCOPE(v0_GRN_validate_and_infer_types);
    const auto& data_pshape = get_input_partial_shape(0);

    if (data_pshape.rank().is_static()) {
        // Input data must be 2, 3 or 4D tensor.
        NODE_VALIDATION_CHECK(this,
                              (data_pshape.size() >= 2 && data_pshape.size() <= 4),
                              "Input tensor rank must be 2, 3 or 4 dimensional (actual input "
                              "shape: ",
                              data_pshape,
                              ").");
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

std::shared_ptr<Node> op::v0::GRN::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_GRN_clone_with_new_inputs);
    if (new_args.size() != 1) {
        OPENVINO_THROW("Incorrect number of new arguments");
    }
    return std::make_shared<GRN>(new_args.at(0), m_bias);
}
}  // namespace ov
