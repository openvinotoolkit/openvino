// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/gather.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v1::Gather);
BWDCMP_RTTI_DEFINITION(op::v8::Gather);

op::v1::Gather::Gather(const Output<Node>& params, const Output<Node>& indices, const Output<Node>& axes)
    : GatherBase(params, indices, axes) {
    constructor_validate_and_infer_types();
}

int64_t ngraph::op::v1::Gather::get_axis() const {
    if (!get_constant_from_source(input_value(2)))
        return AXIS_NOT_SET_VALUE;
    return GatherBase::get_axis();
}

bool ngraph::op::v1::Gather::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Gather_visit_attributes);
    return true;
}

shared_ptr<Node> op::v1::Gather::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::Gather>(new_args.at(0), new_args.at(1), new_args.at(2));
}

BWDCMP_RTTI_DEFINITION(op::v7::Gather);

op::v7::Gather::Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims)
    : GatherBase(data, indices, axis, batch_dims) {
    constructor_validate_and_infer_types();
}

void op::v7::Gather::validate_and_infer_types() {
    OV_OP_SCOPE(v7_Gather_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "Indices element type must be of an integral number type.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number(),
                          "Axis element type must be of an integral number type.");

    op::util::GatherBase::validate_and_infer_types();
}

int64_t op::v7::Gather::get_batch_dims() const {
    if (m_batch_dims < 0 && get_input_partial_shape(1).rank().is_static())
        return m_batch_dims + get_input_partial_shape(1).rank().get_length();
    else
        return m_batch_dims;
}

bool ngraph::op::v7::Gather::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v7_Gather_visit_attributes);
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

shared_ptr<Node> op::v7::Gather::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v7::Gather>(new_args.at(0), new_args.at(1), new_args.at(2), m_batch_dims);
}

op::v8::Gather::Gather(const Output<Node>& data,
                       const Output<Node>& indices,
                       const Output<Node>& axis,
                       const int64_t batch_dims)
    : GatherBase(data, indices, axis, batch_dims) {
    constructor_validate_and_infer_types();
}

void op::v8::Gather::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Gather_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "Indices element type must be of an integral number type.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number(),
                          "Axis element type must be of an integral number type.");

    op::util::GatherBase::validate_and_infer_types();
}

int64_t op::v8::Gather::get_batch_dims() const {
    if (m_batch_dims < 0 && get_input_partial_shape(1).rank().is_static())
        return m_batch_dims + get_input_partial_shape(1).rank().get_length();
    else
        return m_batch_dims;
}

bool ngraph::op::v8::Gather::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_Gather_visit_attributes);
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

shared_ptr<Node> op::v8::Gather::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v8::Gather>(new_args.at(0), new_args.at(1), new_args.at(2), m_batch_dims);
}
