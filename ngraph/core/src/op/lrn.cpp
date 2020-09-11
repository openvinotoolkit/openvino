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

#include "ngraph/op/lrn.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/multiply.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::LRN::type_info;

op::LRN::LRN(const Output<Node>& arg, double alpha, double beta, double bias, size_t size)
    : LRN(arg, op::Constant::create(element::i64, Shape{1}, {1}), alpha, beta, bias, size)
{
    add_provenance_group_member(input_value(1).get_node_shared_ptr());
}

op::LRN::LRN(const Output<Node>& arg,
             const Output<Node>& axes,
             double alpha,
             double beta,
             double bias,
             size_t size)
    : Op({arg, axes})
    , m_alpha(alpha)
    , m_beta(beta)
    , m_bias(bias)
    , m_size(size)
{
    constructor_validate_and_infer_types();
}

AxisSet op::LRN::get_reduction_axes() const
{
    AxisSet axes{1}; // channel axis as default
    auto axes_input_node = input_value(1).get_node_shared_ptr();
    if (auto const_op = as_type_ptr<op::Constant>(axes_input_node))
    {
        axes = const_op->get_axis_set_val();
    }
    return axes;
}

void op::LRN::validate_and_infer_types()
{
    element::Type arg_type = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);

    const PartialShape& input_shape = get_input_partial_shape(0);
    const auto input_shape_rank = input_shape.rank();

    PartialShape axes_shape{PartialShape::dynamic()};
    if (get_input_partial_shape(1).is_static())
    {
        axes_shape = get_input_partial_shape(1);
    }

    auto axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(1),
                          "Input axes must have rank equals 1 (axes_rank: ",
                          axes_rank,
                          ").");

    NODE_VALIDATION_CHECK(
        this,
        axes_shape.is_dynamic() || input_shape_rank.is_dynamic() ||
            axes_shape[0].get_length() <= input_shape_rank.get_length(),
        "Number of elements of axes must be >= 0 and <= argument rank (axes_shape[0]: ",
        axes_shape[0],
        ").");

    if (input_shape_rank.is_static())
    {
        const auto reduction_axes = get_reduction_axes();
        for (auto axis : reduction_axes)
        {
            NODE_VALIDATION_CHECK(this,
                                  axis < input_shape_rank.get_length(),
                                  "Reduction axis (",
                                  axis,
                                  ") is out of bounds ",
                                  "(argument shape: ",
                                  input_shape,
                                  ", reduction axes: ",
                                  reduction_axes,
                                  ")");
        }
    }

    const auto& axes_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axes_type.is_integral_number(),
                          "Axes input must be integral numbers, but are: ",
                          axes_type,
                          ").");
}

bool ngraph::op::v0::LRN::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("alpha", m_alpha);
    visitor.on_attribute("beta", m_beta);
    visitor.on_attribute("bias", m_bias);
    visitor.on_attribute("size", m_size);
    return true;
}

shared_ptr<Node> op::LRN::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<op::LRN>(new_args.at(0), new_args.at(1), m_alpha, m_beta, m_bias, m_size);
}
