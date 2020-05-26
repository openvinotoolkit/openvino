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

#include "ngraph/op/one_hot.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::OneHot::type_info;

op::v0::OneHot::OneHot(const Output<Node>& arg, const PartialShape& shape, size_t one_hot_axis)
    : Op({arg})
    , m_shape(shape)
    , m_one_hot_axis(one_hot_axis)
{
    constructor_validate_and_infer_types();
}

void op::v0::OneHot::validate_and_infer_types()
{
    element::Type arg_et = get_input_element_type(0);
    PartialShape arg_shape = get_input_partial_shape(0);
    Rank arg_rank = arg_shape.rank();

    NODE_VALIDATION_CHECK(this,
                          arg_et.is_dynamic() || arg_et.is_integral(),
                          "Argument does not have integral element type.");

    NODE_VALIDATION_CHECK(
        this, m_shape.rank().is_static(), "Requested result shape has dynamic rank.");

    NODE_VALIDATION_CHECK(this,
                          m_one_hot_axis < m_shape.rank().get_length(),
                          "One-hot axis (",
                          m_one_hot_axis,
                          ") is out of bounds (requested result shape: ",
                          m_shape,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          m_shape[m_one_hot_axis].is_static(),
                          "Requested result shape (",
                          m_shape,
                          ") has dynamic dimension at the one-hot axis ",
                          "(",
                          m_one_hot_axis,
                          ").");

    PartialShape result_shape{m_shape};

    if (arg_rank.is_static())
    {
        std::vector<Dimension> expected_input_dims(m_shape.rank().get_length());
        for (size_t i = 0; i < m_shape.rank().get_length(); i++)
        {
            expected_input_dims[i] = m_shape[i];
        }
        expected_input_dims.erase(expected_input_dims.begin() + m_one_hot_axis);
        PartialShape expected_input_shape{expected_input_dims};

        PartialShape merged_input_shape{expected_input_shape};
        NODE_VALIDATION_CHECK(this,
                              PartialShape::merge_into(merged_input_shape, arg_shape),
                              "Argument shape ",
                              arg_shape,
                              " does not match the expected shape of ",
                              expected_input_shape,
                              ".");

        std::vector<Dimension> output_dims(merged_input_shape.rank().get_length());
        for (size_t i = 0; i < merged_input_shape.rank().get_length(); i++)
        {
            output_dims[i] = merged_input_shape[i];
        }
        output_dims.insert(output_dims.begin() + m_one_hot_axis, m_shape[m_one_hot_axis]);
        result_shape = PartialShape{output_dims};
    }

    set_output_type(0, arg_et, result_shape);
}

shared_ptr<Node> op::v0::OneHot::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v0::OneHot>(new_args.at(0), m_shape, m_one_hot_axis);
}

constexpr NodeTypeInfo op::v1::OneHot::type_info;

op::v1::OneHot::OneHot(const Output<Node>& indices,
                       const Output<Node>& depth,
                       const Output<Node>& on_value,
                       const Output<Node>& off_value,
                       int64_t axis)
    : Op({indices, depth, on_value, off_value})
    , m_axis(axis)
{
    constructor_validate_and_infer_types();
}

void op::v1::OneHot::validate_and_infer_types()
{
    const auto& indices_et = get_input_element_type(0);
    const auto& depth_et = get_input_element_type(1);
    const auto& on_value_et = get_input_element_type(2);
    const auto& off_value_et = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          indices_et.is_dynamic() || indices_et.is_integral(),
                          "Indices must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          depth_et.is_dynamic() || depth_et.is_integral(),
                          "Depth must be integral element type.");

    NODE_VALIDATION_CHECK(this,
                          on_value_et.compatible(off_value_et),
                          "on_value element type must be compatible with off_value element type.");

    const auto& indices_shape = get_input_partial_shape(0);
    const auto& depth_shape = get_input_partial_shape(1);
    const auto& on_value_shape = get_input_partial_shape(2);
    const auto& off_value_shape = get_input_partial_shape(3);

    NODE_VALIDATION_CHECK(this,
                          depth_shape.is_dynamic() || is_scalar(depth_shape.to_shape()),
                          "depth input must be scalar.");

    NODE_VALIDATION_CHECK(this,
                          on_value_shape.is_dynamic() || is_scalar(on_value_shape.to_shape()),
                          "on_value input must be scalar.");

    NODE_VALIDATION_CHECK(this,
                          off_value_shape.is_dynamic() || is_scalar(off_value_shape.to_shape()),
                          "off_value input must be scalar.");

    const auto& depth = input_value(1).get_node_shared_ptr();
    PartialShape result_shape{PartialShape::dynamic()};

    if (indices_shape.is_static() && indices_shape.rank().is_static() && is_type<op::v0::Constant>(depth))
    {
        const auto indices_rank = indices_shape.rank().get_length();

        std::vector<Dimension> out_dims(indices_rank);
        for (auto i = 0; i < indices_rank; i++)
        {
            out_dims[i] = indices_shape[i];
        }
        m_axis =
            ngraph::normalize_axis(this, m_axis, indices_rank + 1, -indices_rank - 1, indices_rank);

        auto depth_element_type = depth->get_output_element_type(0);
        NODE_VALIDATION_CHECK(this,
                              depth_element_type.is_integral(),
                              "'depth' input element type must be an integer (got ",
                              depth_element_type,
                              ").");

        NODE_VALIDATION_CHECK(this,
                              is_scalar(depth->get_shape()),
                              "A scalar input should be provided as 'depth' to OneHot",
                              " (got ",
                              depth->get_shape(),
                              " elements).");

        const auto depth_constant = as_type_ptr<op::Constant>(depth);
        int64_t depth_val = depth_constant->cast_vector<int64_t>()[0];

        NODE_VALIDATION_CHECK(this,
                              depth_val > 0,
                              "The value of 'depth' must be a positive number.",
                              " (got ",
                              depth_val,
                              ").");

        out_dims.insert(out_dims.begin() + m_axis, Dimension(depth_val));
        result_shape = out_dims;
    }

    set_output_type(0, on_value_et, result_shape);
}

bool ngraph::op::v1::OneHot::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("axis", m_axis);
    return true;
}

shared_ptr<Node> op::v1::OneHot::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::OneHot>(
        new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_axis);
}
