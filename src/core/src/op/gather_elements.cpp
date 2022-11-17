// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gather_elements.hpp"

#include <gather_elements_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

// ------------------------------ V6 ------------------------------

BWDCMP_RTTI_DEFINITION(op::v6::GatherElements);

op::v6::GatherElements::GatherElements(const Output<Node>& data, const Output<Node>& indices, const int64_t axis)
    : Op({data, indices}),
      m_axis(axis) {
    constructor_validate_and_infer_types();
}

void op::v6::GatherElements::validate_and_infer_types() {
    OV_OP_SCOPE(v6_GatherElements_validate_and_infer_types);
    const auto& data_type = get_input_element_type(0);
    const auto& indices_type = get_input_element_type(1);

    NODE_VALIDATION_CHECK(this,
                          indices_type == element::Type_t::i32 || indices_type == element::Type_t::i64,
                          "indices must be of int32 or int64 type. But instead got: ",
                          indices_type);

    const auto& data_pshape = get_input_partial_shape(0);
    const auto& indices_pshape = get_input_partial_shape(1);
    std::vector<PartialShape> input_shapes = {data_pshape, indices_pshape}, output_shapes = {PartialShape{}};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, data_type, output_shapes[0]);
}

bool op::v6::GatherElements::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_GatherElements_visit_attributes);
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v6::GatherElements::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_GatherElements_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<op::v6::GatherElements>(new_args.at(0), new_args.at(1), m_axis);
}
