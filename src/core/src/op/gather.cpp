// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/gather.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov {
namespace op {
namespace v1 {
Gather::Gather(const Output<Node>& params, const Output<Node>& indices, const Output<Node>& axes)
    : GatherBase(params, indices, axes) {
    constructor_validate_and_infer_types();
}

int64_t Gather::get_axis() const {
    return ov::util::get_constant_from_source(input_value(2)) ? GatherBase::get_axis() : AXIS_NOT_SET_VALUE;
}

std::shared_ptr<Node> Gather::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Gather>(new_args.at(0), new_args.at(1), new_args.at(2));
}
}  // namespace v1

namespace v7 {
Gather::Gather(const Output<Node>& data,
               const Output<Node>& indices,
               const Output<Node>& axis,
               const int64_t batch_dims)
    : GatherBase(data, indices, axis, batch_dims) {
    constructor_validate_and_infer_types();
}

void Gather::validate_and_infer_types() {
    OV_OP_SCOPE(v7_Gather_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "Indices element type must be of an integral number type.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number(),
                          "Axis element type must be of an integral number type.");

    util::GatherBase::validate_and_infer_types();
}

int64_t Gather::get_batch_dims() const {
    if (m_batch_dims < 0 && get_input_partial_shape(1).rank().is_static())
        return m_batch_dims + get_input_partial_shape(1).rank().get_length();
    else
        return m_batch_dims;
}

bool Gather::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v7_Gather_visit_attributes);
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

std::shared_ptr<Node> Gather::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v7_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Gather>(new_args.at(0), new_args.at(1), new_args.at(2), m_batch_dims);
}
}  // namespace v7

namespace v8 {
Gather::Gather(const Output<Node>& data,
               const Output<Node>& indices,
               const Output<Node>& axis,
               const int64_t batch_dims)
    : GatherBase(data, indices, axis, batch_dims) {
    constructor_validate_and_infer_types();
}

void Gather::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Gather_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(1).is_integral_number(),
                          "Indices element type must be of an integral number type.");

    NODE_VALIDATION_CHECK(this,
                          get_input_element_type(2).is_integral_number(),
                          "Axis element type must be of an integral number type.");

    op::util::GatherBase::validate_and_infer_types();
}

int64_t Gather::get_batch_dims() const {
    if (m_batch_dims < 0 && get_input_partial_shape(1).rank().is_static())
        return m_batch_dims + get_input_partial_shape(1).rank().get_length();
    else
        return m_batch_dims;
}

bool Gather::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v8_Gather_visit_attributes);
    visitor.on_attribute("batch_dims", m_batch_dims);
    return true;
}

std::shared_ptr<Node> Gather::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v8_Gather_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<Gather>(new_args.at(0), new_args.at(1), new_args.at(2), m_batch_dims);
}

}  // namespace v8
}  // namespace op
}  // namespace ov
