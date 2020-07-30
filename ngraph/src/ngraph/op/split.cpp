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
#include <numeric>

#include "ngraph/attribute_visitor.hpp"
#include "ngraph/builder/split.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/split.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::v0::Split::type_info;

op::v0::Split::Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_split)
    : FusedOp({data, axis})
    , m_split_evenly{true}
    , m_num_split{num_split}
{
    constructor_validate_and_infer_types();
}

op::v0::Split::Split(const Output<Node>& data,
                     const Output<Node>& axis,
                     const std::vector<size_t>& splits)
    : FusedOp({data, axis})
    , m_split_evenly{false}
    , m_num_split{0}
    , m_splits{splits}
{
    constructor_validate_and_infer_types();
}

void op::v0::Split::pre_validate_and_infer_types()
{
    const auto axis_shape = get_input_shape(1);
    NODE_VALIDATION_CHECK(this, is_scalar(axis_shape), "The 'axis' input node must be scalar");

    const auto axis_node = input_value(1).get_node_shared_ptr();
    NODE_VALIDATION_CHECK(
        this, op::is_constant(axis_node), "The 'axis' input node must be constant");
    const auto axis_node_const = as_type_ptr<op::Constant>(axis_node);
    m_axis = axis_node_const->get_data_ptr<int64_t>()[0];

    // Create dynamic-typed outputs. Actual shape/type will be computed during shape inference
    for (size_t i = 0; i < std::max(m_splits.size(), m_num_split); i++)
    {
        set_output_type(i, get_input_element_type(0), PartialShape::dynamic());
    }

    if (is_dynamic())
    {
        return;
    }

    const auto shape = get_input_shape(0);

    const auto data_rank = get_input_partial_shape(0).rank();
    m_axis = ngraph::normalize_axis(this, m_axis, data_rank);
    const auto dimension_at_axis = shape.at(m_axis);
    if (m_split_evenly)
    {
        NODE_VALIDATION_CHECK(this,
                              dimension_at_axis % m_num_split == 0,
                              "The input tensor's dimension pointed by the 'axis' parameter: ",
                              dimension_at_axis,
                              " has to be a multiple of the 'num_split' parameter value: ",
                              m_num_split);

        m_splits.assign(m_num_split, dimension_at_axis / m_num_split);
    }
    else
    {
        const auto sum_splits = accumulate(begin(m_splits), end(m_splits), 0UL);
        NODE_VALIDATION_CHECK(this,
                              sum_splits == dimension_at_axis,
                              "The input tensor's dimension pointed by the 'axis' parameter: ",
                              dimension_at_axis,
                              " has to be equal to the sum of splits passed to the op: ",
                              sum_splits);

        const bool all_splits_positive =
            all_of(begin(m_splits), end(m_splits), [](const size_t v) { return v > 0; });

        NODE_VALIDATION_CHECK(this,
                              all_splits_positive == true,
                              "All values of the 'splits' attribute must be greater than zero");
    }
    set_input_is_relevant_to_shape(0);
}

OutputVector op::v0::Split::decompose_op() const
{
    return builder::split(input_value(0), m_splits, m_axis);
}

shared_ptr<Node> op::v0::Split::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<Split>(new_args.at(0), new_args.at(1), m_splits);
}

constexpr NodeTypeInfo op::v1::Split::type_info;

op::v1::Split::Split(const Output<Node>& data, const Output<Node>& axis, const size_t num_splits)
    : Op({data, axis})
    , m_num_splits{num_splits}
{
    constructor_validate_and_infer_types();
}

bool ngraph::op::v1::Split::visit_attributes(AttributeVisitor& visitor)
{
    visitor.on_attribute("num_splits", m_num_splits);
    return true;
}

void op::v1::Split::validate_and_infer_types()
{
    const auto data_ps = input_value(0).get_partial_shape();
    const auto axis_ps = input_value(1).get_partial_shape();
    const auto axis_et = input_value(1).get_element_type();

    NODE_VALIDATION_CHECK(this,
                          axis_ps.rank().is_static() && axis_ps.rank().get_length() == 0,
                          "The 'axis' input is expected to be a scalar. Got: ",
                          axis_ps);

    NODE_VALIDATION_CHECK(
        this, axis_et.is_integral(), "The 'axis' input only accepts integral types");

    if (op::is_constant(input_value(1).get_node()) && data_ps.is_static())
    {
        const auto axis_input = as_type_ptr<op::Constant>(input_value(1).get_node_shared_ptr());
        auto axis = axis_input->cast_vector<int64_t>()[0];

        const auto data_rank = get_input_partial_shape(0).rank();
        axis = ngraph::normalize_axis(this, axis, data_rank);

        const auto data_shape = data_ps.to_shape();
        const auto dimension_at_axis = data_shape.at(axis);

        NODE_VALIDATION_CHECK(this,
                              dimension_at_axis % m_num_splits == 0,
                              "The input tensor's dimension pointed by the 'axis' parameter: ",
                              dimension_at_axis,
                              " has to be a multiple of the 'num_splits' attribute value: ",
                              m_num_splits);

        Shape each_output_shape{data_shape};
        each_output_shape.at(axis) = dimension_at_axis / m_num_splits;

        for (size_t i = 0; i < m_num_splits; ++i)
        {
            set_output_type(i, get_input_element_type(0), each_output_shape);
        }
    }
    else
    {
        for (size_t i = 0; i < m_num_splits; ++i)
        {
            set_output_type(i, get_input_element_type(0), PartialShape::dynamic());
        }

        set_input_is_relevant_to_shape(0);
    }
}

shared_ptr<Node> op::v1::Split::clone_with_new_inputs(const OutputVector& new_args) const
{
    check_new_args_count(this, new_args);
    return make_shared<v1::Split>(new_args.at(0), new_args.at(1), m_num_splits);
}
