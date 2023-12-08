// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/avg_pool.hpp"

#include "avg_pool_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"

// *** AvgPool OP SET 1 ***
ov::op::v1::AvgPool::AvgPool(const Output<Node>& arg,
                             const Strides& strides,
                             const Shape& pads_begin,
                             const Shape& pads_end,
                             const Shape& kernel,
                             bool exclude_pad,
                             op::RoundingType rounding_type,
                             const PadType& auto_pad)
    : Op({arg}),
      m_kernel(kernel),
      m_strides(strides),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_exclude_pad(exclude_pad),
      m_auto_pad(auto_pad),
      m_rounding_type(rounding_type) {
    constructor_validate_and_infer_types();
}

bool ov::op::v1::AvgPool::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_AvgPool_visit_attributes);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("exclude-pad", m_exclude_pad);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("rounding_type", m_rounding_type);
    return true;
}

void ov::op::v1::AvgPool::validate_and_infer_types() {
    OV_OP_SCOPE(v1_AvgPool_validate_and_infer_types);

    const auto output_shapes =
        shape_infer(this, ov::util::get_node_input_partial_shapes(*this), m_pads_begin, m_pads_end);
    set_output_type(0, get_input_element_type(0), output_shapes.front());
}

const ov::Shape& ov::op::v1::AvgPool::get_kernel() const {
    return m_kernel;
}

void ov::op::v1::AvgPool::set_kernel(const Shape& kernel) {
    m_kernel = kernel;
}

const ov::Strides& ov::op::v1::AvgPool::get_strides() const {
    return m_strides;
}

void ov::op::v1::AvgPool::set_strides(const Strides& strides) {
    m_strides = strides;
}

const ov::Shape& ov::op::v1::AvgPool::get_pads_begin() const {
    return m_pads_begin;
}

void ov::op::v1::AvgPool::set_pads_begin(const Shape& pads_begin) {
    m_pads_begin = pads_begin;
}

const ov::Shape& ov::op::v1::AvgPool::get_pads_end() const {
    return m_pads_end;
}

void ov::op::v1::AvgPool::set_pads_end(const Shape& pads_end) {
    m_pads_end = pads_end;
}

bool ov::op::v1::AvgPool::get_exclude_pad() const {
    return m_exclude_pad;
}

void ov::op::v1::AvgPool::set_exclude_pad(bool exclude_pad) {
    m_exclude_pad = exclude_pad;
}

const ov::op::PadType& ov::op::v1::AvgPool::get_auto_pad() const {
    return m_auto_pad;
}

void ov::op::v1::AvgPool::set_auto_pad(const op::PadType& auto_pad) {
    m_auto_pad = auto_pad;
}

ov::op::RoundingType ov::op::v1::AvgPool::get_rounding_type() const {
    return m_rounding_type;
}

void ov::op::v1::AvgPool::set_rounding_type(op::RoundingType rounding_type) {
    m_rounding_type = rounding_type;
}

std::shared_ptr<ov::Node> ov::op::v1::AvgPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_AvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<AvgPool>(new_args.at(0),
                                     m_strides,
                                     m_pads_begin,
                                     m_pads_end,
                                     m_kernel,
                                     m_exclude_pad,
                                     m_rounding_type,
                                     m_auto_pad);
}
