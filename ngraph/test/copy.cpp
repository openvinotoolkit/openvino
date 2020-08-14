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

#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "ngraph/ngraph.hpp"
#include "util/ndarray.hpp"
#include "util/test_tools.hpp"

using namespace std;
using namespace ngraph;

template <typename OP>
bool check_unary()
{
    Shape shape{1};
    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<OP>(arg0);
    auto new_node = node->copy_with_new_inputs(new_args);

    return (nullptr != new_node) && (new_args == new_node->input_values());
}

template <typename OP>
bool check_binary()
{
    Shape shape{1};
    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape);
    auto arg1 = make_shared<op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape),
                          make_shared<op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<OP>(arg0, arg1);
    auto new_node = node->copy_with_new_inputs(new_args);

    return (nullptr != new_node) && (new_args == new_node->input_values());
}

TEST(copy, abs)
{
    ASSERT_TRUE(check_unary<op::v0::Abs>());
}

TEST(copy, acos)
{
    ASSERT_TRUE(check_unary<op::v0::Acos>());
}

TEST(copy, add)
{
    ASSERT_TRUE(check_binary<op::v0::Add>());
}

TEST(copy, asin)
{
    ASSERT_TRUE(check_unary<op::v0::Asin>());
}

TEST(copy, atan)
{
    ASSERT_TRUE(check_unary<op::v0::Atan>());
}

TEST(copy, broadcast)
{
    Shape shape1{1};
    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape1);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape1)};

    Shape shape{4, 1, 3};
    AxisSet axes{0, 2};

    auto node = make_shared<op::v0::Broadcast>(arg0, shape, axes);
    auto new_node = node->copy_with_new_inputs(new_args);
    auto node_cast = as_type_ptr<op::v0::Broadcast>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(shape == node_cast->get_broadcast_shape());
    ASSERT_TRUE(axes == node_cast->get_broadcast_axes());
}

TEST(copy, ceiling)
{
    ASSERT_TRUE(check_unary<op::v0::Ceiling>());
}

TEST(copy, concat)
{
    Shape shape{1};
    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape);
    auto arg1 = make_shared<op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape),
                          make_shared<op::v0::Parameter>(element::f32, shape)};
    size_t axis = 0;
    auto node = make_shared<op::v0::Concat>(NodeVector{arg0, arg1}, axis);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = as_type_ptr<op::v0::Concat>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(node_cast->get_concatenation_axis() == axis);
}

TEST(copy, constant)
{
    Shape shape{};
    vector<float> c{2.4f};
    auto& et = element::f32;
    auto node = op::v0::Constant::create(et, shape, c);
    auto new_node = node->clone_with_new_inputs(OutputVector{});
    auto node_cast = as_type_ptr<op::v0::Constant>(new_node);
    ASSERT_NE(node_cast, nullptr);
    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(OutputVector{} == new_node->input_values());
    ASSERT_TRUE(node_cast->get_vector<float>() == c);
    ASSERT_TRUE(node_cast->get_shape() == shape);
    ASSERT_TRUE(node_cast->get_element_type() == et);
}

TEST(copy, convert)
{
    Shape shape;
    auto& et = element::f64;
    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<op::v0::Convert>(arg0, et);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = as_type_ptr<op::v0::Convert>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(et == node_cast->get_convert_element_type());
}

TEST(copy, cos)
{
    ASSERT_TRUE(check_unary<op::v0::Cos>());
}

TEST(copy, cosh)
{
    ASSERT_TRUE(check_unary<op::v0::Cosh>());
}

TEST(copy, divide)
{
    ASSERT_TRUE(check_binary<op::v0::Divide>());
}

TEST(copy, dot)
{
    ASSERT_TRUE(check_binary<op::v0::Dot>());
}

TEST(copy, equal)
{
    ASSERT_TRUE(check_binary<op::v0::Equal>());
}

TEST(copy, exp)
{
    ASSERT_TRUE(check_unary<op::v0::Exp>());
}

TEST(copy, floor)
{
    ASSERT_TRUE(check_unary<op::v0::Floor>());
}

TEST(copy, greater_eq)
{
    ASSERT_TRUE(check_binary<op::v0::GreaterEq>());
}

TEST(copy, greater)
{
    ASSERT_TRUE(check_binary<op::v0::Greater>());
}

TEST(copy, less_eq)
{
    ASSERT_TRUE(check_binary<op::v0::LessEq>());
}

TEST(copy, less)
{
    ASSERT_TRUE(check_binary<op::v0::Less>());
}

TEST(copy, log)
{
    ASSERT_TRUE(check_unary<op::v0::Log>());
}

TEST(copy, maximum)
{
    ASSERT_TRUE(check_binary<op::v0::Maximum>());
}

TEST(copy, minimum)
{
    ASSERT_TRUE(check_binary<op::v0::Minimum>());
}

TEST(copy, multiply)
{
    ASSERT_TRUE(check_binary<op::v0::Multiply>());
}

TEST(copy, negative)
{
    ASSERT_TRUE(check_unary<op::v0::Negative>());
}

TEST(copy, not_equal)
{
    ASSERT_TRUE(check_binary<op::v0::NotEqual>());
}

TEST(copy, parameter)
{
    Shape shape{1};
    auto node = make_shared<op::v0::Parameter>(element::f32, shape);
    auto new_node = node->clone_with_new_inputs({});
    auto node_cast = as_type_ptr<op::v0::Parameter>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_node->input_values().size() == 0);
    ASSERT_TRUE(node->has_same_type(new_node));
}

TEST(copy, power)
{
    ASSERT_TRUE(check_binary<op::v0::Power>());
}

TEST(copy, reshape)
{
    Shape shape_in{2, 3, 4};
    AxisVector axes{0, 1, 2};
    Shape shape_out{6, 4};

    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape_in);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape_in)};

    auto node = make_shared<op::v0::Reshape>(arg0, axes, shape_out);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = as_type_ptr<op::v0::Reshape>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(axes == node_cast->get_input_order());
    ASSERT_TRUE(shape_out == node_cast->get_output_shape(0));
}

TEST(copy, select)
{
    Shape shape{1};
    auto arg0 = make_shared<op::v0::Parameter>(element::boolean, shape);
    auto arg1 = make_shared<op::v0::Parameter>(element::f32, shape);
    auto arg2 = make_shared<op::v0::Parameter>(element::f32, shape);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::boolean, shape),
                          make_shared<op::v0::Parameter>(element::f32, shape),
                          make_shared<op::v0::Parameter>(element::f32, shape)};

    auto node = make_shared<op::v0::Select>(arg0, arg1, arg2);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = as_type_ptr<op::v0::Select>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
}

TEST(copy, sign)
{
    ASSERT_TRUE(check_unary<op::v0::Sign>());
}

TEST(copy, sin)
{
    ASSERT_TRUE(check_unary<op::v0::Sin>());
}

TEST(copy, sinh)
{
    ASSERT_TRUE(check_unary<op::v0::Sinh>());
}

TEST(copy, slice)
{
    Shape shape_in{2, 3, 4};
    Coordinate lower{0, 0, 0};
    Coordinate upper{2, 3, 4};
    Strides strides{1, 1, 1};

    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape_in);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape_in)};

    auto node = make_shared<op::v0::Slice>(arg0, lower, upper, strides);
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = as_type_ptr<op::v0::Slice>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(lower == node_cast->get_lower_bounds());
    ASSERT_TRUE(upper == node_cast->get_upper_bounds());
    ASSERT_TRUE(strides == node_cast->get_strides());
}

TEST(copy, subtract)
{
    ASSERT_TRUE(check_binary<op::v0::Subtract>());
}

TEST(copy, sum)
{
    Shape shape{4, 3};
    AxisSet axes{1};
    auto arg0 = make_shared<op::v0::Parameter>(element::f32, shape);

    auto node = make_shared<op::v0::Sum>(arg0, axes);
    OutputVector new_args{make_shared<op::v0::Parameter>(element::f32, shape),
                          node->input_value(1).get_node_shared_ptr()};
    auto new_node = node->clone_with_new_inputs(new_args);
    auto node_cast = as_type_ptr<op::v0::Sum>(new_node);
    ASSERT_NE(node_cast, nullptr);

    ASSERT_TRUE(nullptr != new_node);
    ASSERT_TRUE(new_args == new_node->input_values());
    ASSERT_TRUE(axes == node_cast->get_reduction_axes());
}

TEST(copy, tan)
{
    ASSERT_TRUE(check_unary<op::v0::Tan>());
}

TEST(copy, tanh)
{
    ASSERT_TRUE(check_unary<op::v0::Tanh>());
}
