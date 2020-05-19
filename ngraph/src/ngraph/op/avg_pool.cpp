//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/op/avg_pool.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

// *** AvgPool OP SET 0 ***
constexpr NodeTypeInfo op::v0::AvgPool::type_info;

op::v0::AvgPool::AvgPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides,
                         const Shape& padding_below,
                         const Shape& padding_above,
                         bool include_padding_in_avg_computation,
                         const PadType& pad_type,
                         bool ceil_mode)
    : Op({arg})
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
    , m_pad_type(pad_type)
    , m_ceil_mode(ceil_mode)
{
    constructor_validate_and_infer_types();
}

op::v0::AvgPool::AvgPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides,
                         const Shape& padding_below,
                         const Shape& padding_above,
                         bool include_padding_in_avg_computation,
                         const PadType& pad_type)
    : AvgPool(arg,
              window_shape,
              window_movement_strides,
              padding_below,
              padding_above,
              include_padding_in_avg_computation,
              pad_type,
              false)
{
}

op::v0::AvgPool::AvgPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides,
                         const Shape& padding_below,
                         const Shape& padding_above,
                         bool include_padding_in_avg_computation)
    : AvgPool(arg,
              window_shape,
              window_movement_strides,
              padding_below,
              padding_above,
              include_padding_in_avg_computation,
              PadType::EXPLICIT)
{
}

bool op::v0::AvgPool::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("window_shape", m_window_shape);
    visitor.on_attribute("window_movement_strides", m_window_movement_strides);
    visitor.on_attribute("padding_below", m_padding_below);
    visitor.on_attribute("padding_above", m_padding_above);
    visitor.on_attribute("include_padding_in_avg_computation",
                         m_include_padding_in_avg_computation);
    visitor.on_attribute("pad_type", m_pad_type);
    visitor.on_attribute("ceil_mode", m_ceil_mode);
    return true;
}

void op::v0::AvgPool::validate_and_infer_types()
{
    if (0 == m_window_movement_strides.size())
    {
        m_window_movement_strides = Strides(m_window_shape.size(), 1);
    }

    if (0 == m_padding_below.size())
    {
        m_padding_below = Shape(m_window_shape.size(), 0);
    }

    if (0 == m_padding_above.size())
    {
        m_padding_above = Shape(m_window_shape.size(), 0);
    }

    const PartialShape& arg_shape = get_input_partial_shape(0);

    if (m_pad_type == PadType::SAME_UPPER || m_pad_type == PadType::SAME_LOWER)
    {
        if (arg_shape.is_static())
        {
            CoordinateDiff padding_above, padding_below;
            infer_auto_padding(arg_shape.to_shape(),
                               m_window_shape,
                               m_window_movement_strides,
                               Strides(m_window_shape.size(), 1), // No dilation
                               m_pad_type,
                               padding_above,
                               padding_below);
            m_padding_above = Shape(padding_above.begin(), padding_above.end());
            m_padding_below = Shape(padding_below.begin(), padding_below.end());
        }
    }

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    set_output_type(0,
                    get_input_element_type(0),
                    infer_batched_pooling_forward(this,
                                                  arg_shape,
                                                  padding_below,
                                                  padding_above,
                                                  m_window_shape,
                                                  m_window_movement_strides,
                                                  m_include_padding_in_avg_computation,
                                                  m_ceil_mode));
}

op::v0::AvgPool::AvgPool(const Output<Node>& arg,
                         const Shape& window_shape,
                         const Strides& window_movement_strides)
    : AvgPool(arg, window_shape, window_movement_strides, Shape(), Shape(), false)
{
}

op::v0::AvgPool::AvgPool(const Output<Node>& arg, const Shape& window_shape)
    : AvgPool(arg, window_shape, Strides(), Shape(), Shape(), false)
{
}

const Shape& op::v0::AvgPool::get_window_shape() const
{
    return m_window_shape;
}

void op::v0::AvgPool::set_window_shape(const Shape& window_shape)
{
    m_window_shape = window_shape;
}

const Strides& op::v0::AvgPool::get_window_movement_strides() const
{
    return m_window_movement_strides;
}

void op::v0::AvgPool::set_window_movement_strides(const Strides& window_movement_strides)
{
    m_window_movement_strides = window_movement_strides;
}

const Shape& op::v0::AvgPool::get_padding_below() const
{
    return m_padding_below;
}

void op::v0::AvgPool::set_padding_below(const Shape& padding_below)
{
    m_padding_below = padding_below;
}

const Shape& op::v0::AvgPool::get_padding_above() const
{
    return m_padding_above;
}

void op::v0::AvgPool::set_padding_above(const Shape& padding_above)
{
    m_padding_above = padding_above;
}

bool op::v0::AvgPool::get_include_padding_in_avg_computation() const
{
    return m_include_padding_in_avg_computation;
}

void op::v0::AvgPool::set_include_padding_in_avg_computation(
    bool include_padding_in_avg_computation)
{
    m_include_padding_in_avg_computation = include_padding_in_avg_computation;
}

const op::PadType& op::v0::AvgPool::get_pad_type() const
{
    return m_pad_type;
}

void op::v0::AvgPool::set_pad_type(const op::PadType& pad_type)
{
    m_pad_type = pad_type;
}

bool op::v0::AvgPool::get_ceil_mode() const
{
    return m_ceil_mode;
}

void op::v0::AvgPool::set_ceil_mode(bool ceil_mode)
{
    m_ceil_mode = ceil_mode;
}

shared_ptr<Node> op::v0::AvgPool::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::AvgPool>(new_args.at(0),
                                    m_window_shape,
                                    m_window_movement_strides,
                                    m_padding_below,
                                    m_padding_above,
                                    m_include_padding_in_avg_computation,
                                    m_pad_type,
                                    m_ceil_mode);
}

constexpr NodeTypeInfo op::v0::AvgPoolBackprop::type_info;
shared_ptr<Node> op::v0::AvgPool::get_default_value() const
{
    return Constant::create(get_element_type(), get_shape(), {0});
}

op::v0::AvgPoolBackprop::AvgPoolBackprop(const Shape& forward_arg_shape,
                                         const Output<Node>& delta,
                                         const Shape& window_shape,
                                         const Strides& window_movement_strides,
                                         const Shape& padding_below,
                                         const Shape& padding_above,
                                         bool include_padding_in_avg_computation)
    : Op({delta})
    , m_forward_arg_shape(forward_arg_shape)
    , m_window_shape(window_shape)
    , m_window_movement_strides(window_movement_strides)
    , m_padding_below(padding_below)
    , m_padding_above(padding_above)
    , m_include_padding_in_avg_computation(include_padding_in_avg_computation)
{
    constructor_validate_and_infer_types();
}

bool op::v0::AvgPoolBackprop::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("forward_arg_shape", m_forward_arg_shape);
    visitor.on_attribute("window_shape", m_window_shape);
    visitor.on_attribute("window_movement_strides", m_window_movement_strides);
    visitor.on_attribute("padding_below", m_padding_below);
    visitor.on_attribute("padding_above", m_padding_above);
    visitor.on_attribute("include_padding_in_avg_computation",
                         m_include_padding_in_avg_computation);
    return true;
}

void op::v0::AvgPoolBackprop::validate_and_infer_types()
{
    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff padding_below(m_padding_below.begin(), m_padding_below.end());
    CoordinateDiff padding_above(m_padding_above.begin(), m_padding_above.end());

    PartialShape forward_result_shape =
        infer_batched_pooling_forward(this,
                                      m_forward_arg_shape,
                                      padding_below,
                                      padding_above,
                                      m_window_shape,
                                      m_window_movement_strides,
                                      m_include_padding_in_avg_computation);

    const PartialShape& delta_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this,
        forward_result_shape.compatible(delta_shape),
        "Inferred forward output shape does not match delta shape (inferred forward output ",
        "shape: ",
        forward_result_shape,
        ", delta shape: ",
        delta_shape,
        ").");

    // TODO(amprocte): Once m_forward_arg_shape is allowed to be dynamic, we may technically be
    // able to infer some extra information from forward_result_shape that was not present in the
    // forward arg shape---namely batch size and channel count. Merge that info in.
    set_output_type(0, get_input_element_type(0), m_forward_arg_shape);
}

const Shape& op::v0::AvgPoolBackprop::get_forward_arg_shape() const
{
    return m_forward_arg_shape;
}

void op::v0::AvgPoolBackprop::set_forward_arg_shape(const Shape& forward_arg_shape)
{
    m_forward_arg_shape = forward_arg_shape;
}

const Shape& op::v0::AvgPoolBackprop::get_window_shape() const
{
    return m_window_shape;
}

void op::v0::AvgPoolBackprop::set_window_shape(const Shape& window_shape)
{
    m_window_shape = window_shape;
}

const Strides& op::v0::AvgPoolBackprop::get_window_movement_strides() const
{
    return m_window_movement_strides;
}

void op::v0::AvgPoolBackprop::set_window_movement_strides(const Strides& window_movement_strides)
{
    m_window_movement_strides = window_movement_strides;
}

const Shape& op::v0::AvgPoolBackprop::get_padding_below() const
{
    return m_padding_below;
}

void op::v0::AvgPoolBackprop::set_padding_below(const Shape& padding_below)
{
    m_padding_below = padding_below;
}

const Shape& op::v0::AvgPoolBackprop::get_padding_above() const
{
    return m_padding_above;
}

void op::v0::AvgPoolBackprop::set_padding_above(const Shape& padding_above)
{
    m_padding_above = padding_above;
}

bool op::v0::AvgPoolBackprop::get_include_padding_in_avg_computation() const
{
    return m_include_padding_in_avg_computation;
}

void op::v0::AvgPoolBackprop::set_include_padding_in_avg_computation(
    bool include_padding_in_avg_computation)
{
    m_include_padding_in_avg_computation = include_padding_in_avg_computation;
}

shared_ptr<Node> op::v0::AvgPoolBackprop::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::AvgPoolBackprop>(m_forward_arg_shape,
                                            new_args.at(0),
                                            m_window_shape,
                                            m_window_movement_strides,
                                            m_padding_below,
                                            m_padding_above,
                                            m_include_padding_in_avg_computation);
}

void op::v0::AvgPool::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (m_ceil_mode)
    {
        throw ngraph_error("Autodiff not supported on AvgPool with ceil_mode set");
    }

    auto delta = deltas.at(0);

    auto operand = input_value(0);
    auto& operand_shape = get_input_shape(0);
    auto backprop = make_shared<op::v0::AvgPoolBackprop>(operand_shape,
                                                         delta,
                                                         m_window_shape,
                                                         m_window_movement_strides,
                                                         m_padding_below,
                                                         m_padding_above,
                                                         m_include_padding_in_avg_computation);
    adjoints.add_delta(operand, backprop);
}

// *** AvgPool OP SET 1 ***
constexpr NodeTypeInfo op::v1::AvgPool::type_info;

op::v1::AvgPool::AvgPool(const Output<Node>& arg,
                         const Strides& strides,
                         const Shape& pads_begin,
                         const Shape& pads_end,
                         const Shape& kernel,
                         bool exclude_pad,
                         op::RoundingType rounding_type,
                         const PadType& auto_pad)
    : Op({arg})
    , m_kernel(kernel)
    , m_strides(strides)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_exclude_pad(exclude_pad)
    , m_auto_pad(auto_pad)
    , m_rounding_type(rounding_type)
{
    constructor_validate_and_infer_types();
}

op::v1::AvgPool::AvgPool(const Output<Node>& arg,
                         const Strides& strides,
                         const Shape& pads_begin,
                         const Shape& pads_end,
                         const Shape& kernel,
                         bool exclude_pad,
                         op::RoundingType rounding_type)
    : AvgPool(arg,
              strides,
              pads_begin,
              pads_end,
              kernel,
              exclude_pad,
              rounding_type,
              op::PadType::EXPLICIT)
{
}

bool op::v1::AvgPool::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("exclude_pad", m_exclude_pad);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("rounding_type", m_rounding_type);
    return true;
}

void op::v1::AvgPool::validate_and_infer_types()
{
    if (0 == m_strides.size())
    {
        m_strides = Strides(m_kernel.size(), 1);
    }

    if (0 == m_pads_begin.size())
    {
        m_pads_begin = Shape(m_kernel.size(), 0);
    }

    if (0 == m_pads_end.size())
    {
        m_pads_end = Shape(m_kernel.size(), 0);
    }

    const PartialShape& arg_shape = get_input_partial_shape(0);

    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER)
    {
        if (arg_shape.is_static())
        {
            CoordinateDiff pads_end, pads_begin;
            infer_auto_padding(arg_shape.to_shape(),
                               m_kernel,
                               m_strides,
                               Strides(m_kernel.size(), 1), // No dilation
                               m_auto_pad,
                               pads_end,
                               pads_begin);
            m_pads_end = Shape(pads_end.begin(), pads_end.end());
            m_pads_begin = Shape(pads_begin.begin(), pads_begin.end());
        }
    }

    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
    CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());

    set_output_type(0,
                    get_input_element_type(0),
                    infer_batched_pooling_forward(this,
                                                  arg_shape,
                                                  pads_begin,
                                                  pads_end,
                                                  m_kernel,
                                                  m_strides,
                                                  !m_exclude_pad,
                                                  m_rounding_type == op::RoundingType::CEIL));
}

const Shape& op::v1::AvgPool::get_kernel() const
{
    return m_kernel;
}

void op::v1::AvgPool::set_kernel(const Shape& kernel)
{
    m_kernel = kernel;
}

const Strides& op::v1::AvgPool::get_strides() const
{
    return m_strides;
}

void op::v1::AvgPool::set_strides(const Strides& strides)
{
    m_strides = strides;
}

const Shape& op::v1::AvgPool::get_pads_begin() const
{
    return m_pads_begin;
}

void op::v1::AvgPool::set_pads_begin(const Shape& pads_begin)
{
    m_pads_begin = pads_begin;
}

const Shape& op::v1::AvgPool::get_pads_end() const
{
    return m_pads_end;
}

void op::v1::AvgPool::set_pads_end(const Shape& pads_end)
{
    m_pads_end = pads_end;
}

bool op::v1::AvgPool::get_exclude_pad() const
{
    return m_exclude_pad;
}

void op::v1::AvgPool::set_exclude_pad(bool exclude_pad)
{
    m_exclude_pad = exclude_pad;
}

const op::PadType& op::v1::AvgPool::get_auto_pad() const
{
    return m_auto_pad;
}

void op::v1::AvgPool::set_auto_pad(const op::PadType& auto_pad)
{
    m_auto_pad = auto_pad;
}

op::RoundingType op::v1::AvgPool::get_rounding_type() const
{
    return m_rounding_type;
}

void op::v1::AvgPool::set_rounding_type(op::RoundingType rounding_type)
{
    m_rounding_type = rounding_type;
}

shared_ptr<Node> op::v1::AvgPool::clone_with_new_inputs(const OutputVector& new_args) const
{
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

constexpr NodeTypeInfo op::v1::AvgPoolBackprop::type_info;

op::v1::AvgPoolBackprop::AvgPoolBackprop(const Output<Node>& delta,
                                         const Output<Node>& forward_arg_shape,
                                         const Strides& strides,
                                         const Shape& pads_begin,
                                         const Shape& pads_end,
                                         const Shape& kernel,
                                         bool exclude_pad)
    : Op({delta, forward_arg_shape})
    , m_kernel(kernel)
    , m_strides(strides)
    , m_pads_begin(pads_begin)
    , m_pads_end(pads_end)
    , m_exclude_pad(exclude_pad)
{
    constructor_validate_and_infer_types();
}

bool op::v1::AvgPoolBackprop::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("exclude_pad", m_exclude_pad);
    return true;
}

const Shape op::v1::AvgPoolBackprop::get_forward_arg_shape() const
{
    Shape shape;
    if (auto const_op = as_type<op::Constant>(input_value(1).get_node()))
    {
        shape = const_op->get_shape_val();
    }
    return shape;
}

void op::v1::AvgPoolBackprop::validate_and_infer_types()
{
    // infer_batched_forward_pooling wants CoordinateDiffs for these, while the pooling ops for
    // now still take Shape (no negative padding).
    CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
    CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());

    PartialShape forward_arg_shape{PartialShape::dynamic()};

    if (input_value(1).get_node_shared_ptr()->is_constant())
    {
        forward_arg_shape = get_forward_arg_shape();
    }

    PartialShape forward_result_shape = infer_batched_pooling_forward(
        this, forward_arg_shape, pads_begin, pads_end, m_kernel, m_strides, m_exclude_pad);

    const PartialShape& delta_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this,
        forward_result_shape.compatible(delta_shape),
        "Inferred forward output shape does not match delta shape (inferred forward output ",
        "shape: ",
        forward_result_shape,
        ", delta shape: ",
        delta_shape,
        ").");

    set_input_is_relevant_to_shape(1);
    set_output_type(0, get_input_element_type(0), forward_arg_shape);
}

const Shape& op::v1::AvgPoolBackprop::get_kernel() const
{
    return m_kernel;
}

void op::v1::AvgPoolBackprop::set_kernel(const Shape& kernel)
{
    m_kernel = kernel;
}

const Strides& op::v1::AvgPoolBackprop::get_strides() const
{
    return m_strides;
}

void op::v1::AvgPoolBackprop::set_strides(const Strides& strides)
{
    m_strides = strides;
}

const Shape& op::v1::AvgPoolBackprop::get_pads_begin() const
{
    return m_pads_begin;
}

void op::v1::AvgPoolBackprop::set_pads_begin(const Shape& pads_begin)
{
    m_pads_begin = pads_begin;
}

const Shape& op::v1::AvgPoolBackprop::get_pads_end() const
{
    return m_pads_end;
}

void op::v1::AvgPoolBackprop::set_pads_end(const Shape& pads_end)
{
    m_pads_end = pads_end;
}

bool op::v1::AvgPoolBackprop::get_exclude_pad() const
{
    return m_exclude_pad;
}

void op::v1::AvgPoolBackprop::set_exclude_pad(bool exclude_pad)
{
    m_exclude_pad = exclude_pad;
}

shared_ptr<Node> op::v1::AvgPoolBackprop::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::AvgPoolBackprop>(new_args.at(0),
                                            new_args.at(1),
                                            m_strides,
                                            m_pads_begin,
                                            m_pads_end,
                                            m_kernel,
                                            m_exclude_pad);
}

void op::v1::AvgPool::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    if (m_rounding_type == op::RoundingType::CEIL)
    {
        throw ngraph_error("Autodiff not supported on AvgPool with ceil_mode set");
    }

    auto delta = deltas.at(0);

    auto operand = input_value(0);
    auto backprop = make_shared<op::v1::AvgPoolBackprop>(
        delta, input_value(1), m_strides, m_pads_begin, m_pads_end, m_kernel, m_exclude_pad);
    adjoints.add_delta(operand, backprop);
}

shared_ptr<Node> op::v1::AvgPool::get_default_value() const
{
    return op::Constant::create(get_element_type(), get_shape(), {0});
}
