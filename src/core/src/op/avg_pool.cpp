// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/avg_pool.hpp"

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;

// *** AvgPool OP SET 1 ***
BWDCMP_RTTI_DEFINITION(ov::op::v1::AvgPool);

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
    if (0 == m_strides.size()) {
        m_strides = Strides(m_kernel.size(), 1);
    }

    if (0 == m_pads_begin.size()) {
        m_pads_begin = Shape(m_kernel.size(), 0);
    }

    if (0 == m_pads_end.size()) {
        m_pads_end = Shape(m_kernel.size(), 0);
    }

    const ov::PartialShape& arg_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this,
        arg_shape.rank().compatible(3) || arg_shape.rank().compatible(4) || arg_shape.rank().compatible(5),
        "Expected a 3D, 4D or 5D tensor for the input. Got: ",
        arg_shape);

    if (arg_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_end.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected pads_end size to be equal to input size - 2. Got: ",
                              m_pads_end.size());

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_begin.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected pads_begin size to be equal to input size - 2. Got: ",
                              m_pads_begin.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_kernel.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected kernel size to be equal to input size - 2. Got: ",
                              m_kernel.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_strides.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected strides size to be equal to input size - 2. Got: ",
                              m_kernel.size());
    }

    auto output_shape = ov::PartialShape::dynamic();
    if (arg_shape.rank().is_static() && arg_shape.rank().get_max_length() > 0) {
        output_shape = std::vector<Dimension>(arg_shape.rank().get_max_length(), Dimension::dynamic());
        if (arg_shape[0].is_static()) {
            output_shape[0] = arg_shape[0];  // batch size
        }
        if (arg_shape[1].is_static()) {
            output_shape[1] = arg_shape[1];  // channel size
        }
    }
    bool update_auto_padding_succeed = true;
    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER) {
        CoordinateDiff pads_end;
        CoordinateDiff pads_begin;
        update_auto_padding_succeed = ngraph::try_apply_auto_padding(arg_shape,
                                                                     m_kernel,
                                                                     m_strides,
                                                                     Strides(m_kernel.size(), 1),  // No dilation
                                                                     m_auto_pad,
                                                                     pads_end,
                                                                     pads_begin);
        m_pads_end = Shape(pads_end.begin(), pads_end.end());
        m_pads_begin = Shape(pads_begin.begin(), pads_begin.end());
    }
    if (m_auto_pad == PadType::VALID) {
        m_pads_end = Shape(m_pads_end.size(), 0);
        m_pads_begin = Shape(m_pads_begin.size(), 0);
    }
    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
    CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());
    set_output_type(0,
                    get_input_element_type(0),
                    update_auto_padding_succeed
                        ? ngraph::infer_batched_pooling_forward(this,
                                                                arg_shape,
                                                                pads_begin,
                                                                pads_end,
                                                                m_kernel,
                                                                m_strides,
                                                                !m_exclude_pad,
                                                                m_rounding_type == op::RoundingType::CEIL,
                                                                Strides{})  // no dilation of the window
                        : output_shape);
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

shared_ptr<ov::Node> ov::op::v1::AvgPool::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_AvgPool_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<v1::AvgPool>(new_args.at(0),
                                    m_strides,
                                    m_pads_begin,
                                    m_pads_end,
                                    m_kernel,
                                    m_exclude_pad,
                                    m_rounding_type,
                                    m_auto_pad);
}

shared_ptr<ov::Node> ov::op::v1::AvgPool::get_default_value() const {
    return op::v0::Constant::create(get_element_type(), get_shape(), {0});
}
