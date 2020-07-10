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

#include <functional>
#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/graph_util.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/shape.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::Dot::type_info;

op::Dot::Dot(const Output<Node>& arg0, const Output<Node>& arg1)
    : Dot(arg0, arg1, 0, false)
{
}

op::Dot::Dot(const Output<Node>& arg0,
             const Output<Node>& arg1,
             size_t reduction_axes_count,
             bool has_reduction_axes_count)
    : Op({arg0, arg1})
    , m_reduction_axes_count(reduction_axes_count)
    , m_has_reduction_axes_count(has_reduction_axes_count)
{
    constructor_validate_and_infer_types();
}

void op::Dot::validate_and_infer_types()
{
    element::Type result_et;

    NODE_VALIDATION_CHECK(
        this,
        element::Type::merge(result_et, get_input_element_type(0), get_input_element_type(1)),
        "Arguments do not have the same element type (arg0 element type: ",
        get_input_element_type(0),
        ", arg1 element type: ",
        get_input_element_type(1),
        ").");

    const PartialShape& arg0_shape = get_input_partial_shape(0);
    const PartialShape& arg1_shape = get_input_partial_shape(1);

    // If an explicit value was not passed for reduction axis count at construction time, we have
    // some extra work to do.
    //
    // - If one of the arguments is known to be scalar, the count is 0.
    // - If both of the arguments are known to be nonscalar, the count is 1.
    // - Otherwise, the count is unknown.
    bool reduction_axes_ambiguous = !m_has_reduction_axes_count;

    if (reduction_axes_ambiguous)
    {
        if (arg0_shape.rank().same_scheme(0) || arg1_shape.rank().same_scheme(0))
        {
            m_reduction_axes_count = 0;
            reduction_axes_ambiguous = false;
        }
        else if (arg0_shape.rank().is_static() && arg1_shape.rank().is_static())
        {
            m_reduction_axes_count = 1;
            reduction_axes_ambiguous = false;
        }
    }

    PartialShape result_shape;

    NODE_VALIDATION_CHECK(this,
                          reduction_axes_ambiguous || arg0_shape.rank().is_dynamic() ||
                              m_reduction_axes_count <= arg0_shape.rank().get_length(),
                          "Reduction axes count (",
                          m_reduction_axes_count,
                          ") is too large (arg0 shape: ",
                          arg0_shape,
                          ", arg1 shape: ",
                          arg1_shape,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          reduction_axes_ambiguous || arg1_shape.rank().is_dynamic() ||
                              m_reduction_axes_count <= arg1_shape.rank().get_length(),
                          "Reduction axes count (",
                          m_reduction_axes_count,
                          ") is too large (arg0 shape: ",
                          arg0_shape,
                          ", arg1 shape: ",
                          arg1_shape,
                          ").");

    if (!reduction_axes_ambiguous && arg0_shape.rank().is_static() && arg1_shape.rank().is_static())
    {
        for (size_t i = 0; i < m_reduction_axes_count; i++)
        {
            size_t axis_index_arg0 = arg0_shape.rank().get_length() - m_reduction_axes_count + i;
            size_t axis_index_arg1 = i;

            NODE_VALIDATION_CHECK(
                this,
                arg0_shape[axis_index_arg0].compatible(arg1_shape[axis_index_arg1]),
                "Paired axes (axis ",
                axis_index_arg0,
                " from arg0, axis ",
                axis_index_arg1,
                " from arg1) do not have same length (arg0 shape: ",
                arg0_shape,
                ", arg1 shape: ",
                arg1_shape,
                ", reduction axes count: ",
                m_reduction_axes_count,
                ").");
        }

        std::vector<Dimension> result_dims(arg0_shape.rank().get_length() +
                                           arg1_shape.rank().get_length() -
                                           2 * m_reduction_axes_count);

        size_t i = 0;

        for (size_t j = 0; j < arg0_shape.rank().get_length() - m_reduction_axes_count; j++)
        {
            result_dims[i++] = arg0_shape[j];
        }
        for (size_t j = m_reduction_axes_count; j < arg1_shape.rank().get_length(); j++)
        {
            result_dims[i++] = arg1_shape[j];
        }

        result_shape = PartialShape(result_dims);
    }
    else
    {
        result_shape = PartialShape::dynamic();
    }

    set_output_type(0, result_et, result_shape);
}

shared_ptr<op::Reshape> make_reshape_axes_to_front(const Output<Node>& n,
                                                   const Shape& front_shape,
                                                   const Shape& back_shape)
{
    AxisVector input_order;
    Shape output_shape;

    for (size_t i = 0; i < back_shape.size(); i++)
    {
        input_order.push_back(front_shape.size() + i);
        output_shape.push_back(back_shape[i]);
    }

    for (size_t i = 0; i < front_shape.size(); i++)
    {
        input_order.push_back(i);
        output_shape.push_back(front_shape[i]);
    }

    return make_shared<op::Reshape>(n, input_order, output_shape);
}

shared_ptr<Node> op::Dot::get_default_value() const
{
    return ngraph::make_constant_from_string("0", get_element_type(), get_shape());
}
