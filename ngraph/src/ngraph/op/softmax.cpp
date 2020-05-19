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

#include "ngraph/op/softmax.hpp"

#include <algorithm>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/autobroadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/subtract.hpp"
#include "ngraph/op/sum.hpp"
#include "ngraph/runtime/reference/softmax.hpp"
#include "ngraph/util.hpp"

using namespace std;
using namespace ngraph;

// *** SOFTMAX OP SET 0 ***
constexpr NodeTypeInfo op::v0::Softmax::type_info;

op::v0::Softmax::Softmax(const Output<Node>& arg, const AxisSet& axes)
    : Op({arg})
{
    set_argument(
        1,
        op::Constant::create(element::i64, Shape{axes.to_vector().size()}, axes.to_vector())
            ->output(0));
    add_provenance_group_member(input_value(1).get_node_shared_ptr());
    constructor_validate_and_infer_types();
}

op::v0::Softmax::Softmax(const Output<Node>& arg, const Output<Node>& axes)
    : Op({arg, axes})
{
    constructor_validate_and_infer_types();
}

bool op::v0::Softmax::are_axes_constant() const
{
    return input_value(1).get_node_shared_ptr()->is_constant();
}

const AxisSet op::v0::Softmax::get_axes() const
{
    AxisSet axes;
    auto const_op = dynamic_pointer_cast<op::Constant>(input_value(1).get_node_shared_ptr());
    if (const_op)
    {
        axes = const_op->get_axis_set_val();
    }
    else
    {
        throw ngraph_error("get_axes called on a Softmax node whose 'axes' input is not constant");
    }
    return axes;
}

void op::v0::Softmax::set_axes(const AxisSet& axes)
{
    shared_ptr<Node> current_const = input_value(1).get_node_shared_ptr();
    shared_ptr<Node> replacement_const =
        op::Constant::create(element::i64, Shape{axes.to_vector().size()}, axes.to_vector());
    this->input(1).replace_source_output(replacement_const->output(0));
    replace_provenance_group_member(current_const, replacement_const);
}

void op::v0::Softmax::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);

    if (input_shape.is_dynamic())
    {
        set_output_type(0, get_input_element_type(0), input_shape);
    }
    else
    {
        set_output_type(0, get_input_element_type(0), input_shape.to_shape());

        if (are_axes_constant())
        {
            auto m_axes = get_axes();
            for (auto axis : m_axes)
            {
                NODE_VALIDATION_CHECK(this,
                                      axis < input_shape.rank().get_length(),
                                      "Reduction axis (",
                                      axis,
                                      ") is out of bounds (argument shape: ",
                                      input_shape,
                                      ").");
            }
            // empty axes == all axes
            if (m_axes.size() == 0)
            {
                for (size_t i = 0; i < get_shape().size(); ++i)
                {
                    m_axes.insert(i);
                }
                set_axes(m_axes);
            }
        }
    }

    set_input_is_relevant_to_shape(1);
}

shared_ptr<Node> op::v0::Softmax::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Softmax>(new_args.at(0), new_args.at(1));
}

void op::v0::Softmax::generate_adjoints(autodiff::Adjoints& adjoints, const OutputVector& deltas)
{
    auto delta = deltas.at(0);
    NGRAPH_CHECK(are_axes_constant(), "axes need to be constant");
    auto axes = get_axes();

    auto z = delta * shared_from_this();
    auto zsum = make_shared<op::Sum>(z, axes);

    Shape shape;
    for (size_t i = 0; i < get_shape().size(); ++i)
    {
        if (axes.find(i) == axes.end())
        {
            shape.push_back(get_shape()[i]);
        }
        else
        {
            shape.push_back(1);
        }
    }
    auto order = ngraph::get_default_order(zsum->get_shape());
    auto zreshape = make_shared<op::Reshape>(zsum, order, shape);

    auto adjoint = z - builder::make_with_numpy_broadcast<op::Multiply>(output(0), zreshape);

    auto x = input_value(0);
    adjoints.add_delta(x, adjoint);
}

namespace
{
    template <element::Type_t ET>
    inline bool try_evaluate_softmax(const HostTensorPtr& arg,
                                     const HostTensorPtr& out,
                                     const Shape& shape,
                                     const AxisSet& axes)
    {
        return (ET == arg->get_element_type()) &&
               (runtime::reference::softmax(
                    arg->get_data_ptr<ET>(), out->get_data_ptr<ET>(), shape, axes),
                true);
    }

    bool evaluate_softmax(const HostTensorPtr& arg, const HostTensorPtr& out, const AxisSet& axes)
    {
        auto shape = out->get_shape();
        return try_evaluate_softmax<element::Type_t::f32>(arg, out, shape, axes) ||
               try_evaluate_softmax<element::Type_t::f64>(arg, out, shape, axes);
    }
}

bool op::v0::Softmax::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    outputs[0]->set_unary(inputs[0]);
    return evaluate_softmax(inputs[0], outputs[0], get_axes());
}

// *** SOFTMAX OP SET V1 ***
constexpr NodeTypeInfo op::v1::Softmax::type_info;

op::v1::Softmax::Softmax(const Output<Node>& arg, const size_t axis)
    : Op({arg})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Softmax::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    return true;
}

void op::v1::Softmax::validate_and_infer_types()
{
    const PartialShape& input_shape = get_input_partial_shape(0);
    if (input_shape.rank().is_static())
        NODE_VALIDATION_CHECK(this,
                              m_axis < input_shape.rank().get_length(),
                              "Reduction axis (",
                              m_axis,
                              ") is out of bounds (argument shape: ",
                              input_shape,
                              ").");
    if (input_shape.is_static())
        set_output_type(0, get_input_element_type(0), input_shape.to_shape());
    else
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
}

shared_ptr<Node> op::v1::Softmax::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::v1::Softmax>(new_args.at(0), m_axis);
}

bool op::v1::Softmax::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs)
{
    outputs[0]->set_unary(inputs[0]);
    return evaluate_softmax(inputs[0], outputs[0], AxisSet{m_axis});
}

void op::v1::Softmax::generate_adjoints(autodiff::Adjoints& /* adjoints */,
                                        const OutputVector& /* deltas */)
{
    throw ngraph_error("op::v1::Softmax::generate_adjoints function is not implemented yet");

    /* This might work, but as of this writing we have no way to test it, so we are being
    careful
    auto delta = deltas.at(0);

    auto z = delta * shared_from_this();

    std::vector<size_t> axes(get_shape().size() - m_axis);
    std::iota(std::begin(axes), std::end(axes), m_axis);
    AxisSet axes_set{axes};

    auto zsum = make_shared<op::Sum>(z, axes_set);

    Shape shape;
    for (size_t i = 0; i < get_shape().size(); ++i)
    {
        if (axes_set.find(i) == axes_set.end())
        {
            shape.push_back(get_shape()[i]);
        }
        else
        {
            shape.push_back(1);
        }
    }
    auto order = ngraph::get_default_order(zsum->get_shape());
    auto zreshape = make_shared<op::Reshape>(zsum, order, shape);

    auto adjoint = z - builder::make_with_numpy_broadcast<op::Multiply>(output(0), zreshape);

    auto x = input_value(0);
    adjoints.add_delta(x, adjoint);
    */
}
