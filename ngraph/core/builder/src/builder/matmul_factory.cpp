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

#include <cstddef>
#include <iterator>
#include <memory>
#include <numeric>

#include "builder/autobroadcast.hpp"
#include "builder/make_constant.hpp"
#include "builder/matmul_factory.hpp"
#include "builder/reshape.hpp"
#include "ngraph/op/concat.hpp"
#include "ngraph/op/dot.hpp"
#include "ngraph/op/quantized_dot.hpp"
#include "ngraph/op/reshape.hpp"
#include "ngraph/op/slice.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace ngraph;
using namespace std;

/// \brief      Slice the sub matrix from the input tensor.
///
/// \param[in]  node  The input tensor. Must be at most of rank 3.
/// \param[in]  idx   The index on the first axis, at which to slice sub-matrix.
///
/// \return     The node representing sub matrix.
///
static Output<Node> get_sub_matrix(const Output<Node>& node, size_t idx)
{
    const Shape& shape{node.get_shape()};
    if (shape.size() < 3)
    {
        return node.get_node_shared_ptr();
    }
    // Below bounds defines the sub_matrix through ranges for each input node axis.
    Coordinate lower_bounds(shape.size());
    Coordinate upper_bounds = shape;
    // We assume `node` tensor is of rank equal 3, thus we slice the sub-matrix lying in the last
    // two dimensions at index `idx` of first axis.
    lower_bounds.at(0) = idx;
    upper_bounds.at(0) = idx + 1;

    auto sub_matrix = Output<Node>{make_shared<op::Slice>(node, lower_bounds, upper_bounds)};
    // Remove first single entry dim.
    return builder::opset1::squeeze(sub_matrix);
}

Output<Node> builder::MatmulFactory::get_left()
{
    return m_inputs.at(0);
}

Output<Node> builder::MatmulFactory::get_right()
{
    return m_inputs.at(1);
}

OutputVector builder::MatmulFactory::make_matmul_op()
{
    auto left = get_left();
    auto right = get_right();

    size_t left_rank{left.get_shape().size()};
    size_t right_rank{right.get_shape().size()};

    // First (easy) case that is already internally handled by Ngraph Dot operator.
    // Multiply two tensors where both of them has rank lower equal 2.
    if (left_rank <= 2 && right_rank <= 2)
    {
        return {make_dot(left, right)
                    .get_node_shared_ptr()
                    ->add_provenance_group_members_above(m_inputs)};
    }

    // Second case:
    // Multiply two tensors where at least one of them is rank greater equal 3.

    // Broadcast input arguments only if both of them are not vectors.
    if (left_rank > 1 && right_rank > 1)
    {
        const OutputVector& broadcasted_nodes =
            builder::numpy_broadcast_for_matmul_operation(left, right);

        left = broadcasted_nodes.at(0);
        right = broadcasted_nodes.at(1);
    }
    const auto& left_shape = left.get_shape();
    const auto& right_shape = right.get_shape();

    // Collapse both tensors _stack of matrices_ axes (all except the last two).
    // This will make easier further dot product calculations.
    if (left_shape.size() > 3)
    {
        left = builder::opset1::collapse(left, 0, left_shape.size() - 3);
    }
    if (right_shape.size() > 3)
    {
        right = builder::opset1::collapse(right, 0, right_shape.size() - 3);
    }

    // Perform multiple small dot products
    size_t groups = left.get_shape().at(0);
    // If we haven't broadcast earlier this means that one of the inputs is a vector,
    // thus the number of groups is defined by the shape of the bigger tensor.
    if (right.get_shape().size() > left.get_shape().size())
    {
        groups = right.get_shape().at(0);
    }
    NodeVector small_dots(groups);

    for (size_t g = 0; g < groups; ++g)
    {
        const auto sliced_left = get_sub_matrix(left, g);
        const auto sliced_right = get_sub_matrix(right, g);
        auto sub_dot = make_dot(sliced_left, sliced_right);

        // Expand sub_dot result with single empty outermost axis, in order to
        // later concatenate sub_dots at this axis.
        small_dots.at(g) = builder::opset1::expand_dims(sub_dot);
    }

    // Concatenate sub_dots on groups axis.
    auto result = make_shared<op::Concat>(small_dots, 0);

    if (left_shape.size() <= 3 && right_shape.size() <= 3)
    {
        return {result->add_provenance_group_members_above(m_inputs)};
    }
    // Expand result _stack of matrices_ axes to get expected result shape.
    else
    {
        const Shape& shape{result->get_shape()};
        Shape result_shape(next(begin(shape)), end(shape));
        result_shape.insert(
            begin(result_shape), begin(left_shape), next(begin(left_shape), left_shape.size() - 2));
        return {make_shared<op::Reshape>(result, get_default_order(shape.size()), result_shape)
                    ->add_provenance_group_members_above(m_inputs)};
    }
}

Output<Node> builder::MatmulFactory::make_dot(const Output<Node>& left, const Output<Node>& right)
{
    return make_shared<op::Dot>(left, right);
}

Output<Node> builder::QLinearMatmulFactory::get_right()
{
    return m_inputs.at(3);
}

Output<Node> builder::QLinearMatmulFactory::make_dot(const Output<Node>& left,
                                                     const Output<Node>& right)
{
    ngraph::element::Type output_type;

    if (left.get_element_type() == ngraph::element::u8 &&
        right.get_element_type() == ngraph::element::i8)
    {
        output_type = ngraph::element::i8;
    }
    else if (left.get_element_type() == ngraph::element::u8 &&
             right.get_element_type() == ngraph::element::u8)
    {
        output_type = ngraph::element::u8;
    }

    return std::make_shared<ngraph::op::QuantizedDot>(left,
                                                      right,
                                                      1,
                                                      m_inputs.at(1),
                                                      m_inputs.at(2),
                                                      m_inputs.at(4),
                                                      m_inputs.at(5),
                                                      m_inputs.at(6),
                                                      m_inputs.at(7),
                                                      output_type,
                                                      ngraph::AxisSet{},
                                                      ngraph::AxisSet{},
                                                      ngraph::AxisSet{});
}

Output<Node> builder::MatmulIntegerFactory::make_dot(const Output<Node>& left,
                                                     const Output<Node>& right)
{
    auto num_inputs = m_inputs.size();
    auto scale_one = ngraph::builder::make_constant(ngraph::element::f32, Shape{}, 1);
    auto output_zero_point = ngraph::builder::make_constant(ngraph::element::i32, Shape{}, 0);
    auto left_zero_point = ngraph::builder::make_constant(left.get_element_type(), Shape{}, 0);
    auto right_zero_point = ngraph::builder::make_constant(right.get_element_type(), Shape{}, 0);
    if (num_inputs == 2)
    {
        return std::make_shared<ngraph::op::QuantizedDot>(left,
                                                          right,
                                                          1,
                                                          scale_one,
                                                          left_zero_point,
                                                          scale_one,
                                                          right_zero_point,
                                                          scale_one,
                                                          output_zero_point,
                                                          ngraph::element::i32,
                                                          ngraph::AxisSet{},
                                                          ngraph::AxisSet{},
                                                          ngraph::AxisSet{});
    }

    left_zero_point = m_inputs.at(2).get_node_shared_ptr();
    if (num_inputs == 4)
    {
        right_zero_point = m_inputs.at(3).get_node_shared_ptr();
    }

    return std::make_shared<ngraph::op::QuantizedDot>(left,
                                                      right,
                                                      1,
                                                      scale_one,
                                                      left_zero_point,
                                                      scale_one,
                                                      right_zero_point,
                                                      scale_one,
                                                      output_zero_point,
                                                      ngraph::element::i32,
                                                      ngraph::AxisSet{},
                                                      ngraph::AxisSet{},
                                                      ngraph::AxisSet{});
}
