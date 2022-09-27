// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/grn.hpp"

#include <algorithm>
#include <iterator>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/axis_set.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::GRN);

op::v0::GRN::GRN(const Output<Node>& data, float bias) : Op({data}), m_bias(bias) {
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

    if (data_pshape.is_static()) {
        const ov::Shape& data_shape{data_pshape.to_shape()};

        // Input data must be 2, 3 or 4D tensor.
        NODE_VALIDATION_CHECK(this,
                              (data_shape.size() >= 2 && data_shape.size() <= 4),
                              "Input tensor rank must be 2, 3 or 4 dimensional (actual input "
                              "shape: ",
                              data_shape,
                              ").");
    }
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

shared_ptr<Node> op::v0::GRN::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_GRN_clone_with_new_inputs);
    if (new_args.size() != 1) {
        throw ngraph_error("Incorrect number of new arguments");
    }
    return make_shared<GRN>(new_args.at(0), m_bias);
}
