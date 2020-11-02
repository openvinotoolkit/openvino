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

#include "avg_pool.hpp"
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

shared_ptr<Node> op::v0::AvgPool::get_default_value() const
{
    return Constant::create(get_element_type(), get_shape(), {0});
}
