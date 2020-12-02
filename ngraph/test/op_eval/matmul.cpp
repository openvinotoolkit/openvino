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
#include <vector>

#include "gtest/gtest.h"

#include "ngraph/op/matmul.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "util/all_close_f.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, matmul_dynamic_1D_arg)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::i32, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::i32, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, false, false);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // inner vector contains shapes for arg0, arg1, matmul expected result
    std::vector<std::vector<Shape>> shapes{{Shape{2}, Shape{2}, Shape{}},
                                           {Shape{3}, Shape{2, 3, 2}, Shape{2, 2}},
                                           {Shape{2, 2, 3}, Shape{3}, Shape{2, 2}}};

    std::vector<std::vector<int32_t>> arg0_inputs{
        {2, 3}, {10, 11, 12}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}};
    std::vector<std::vector<int32_t>> arg1_inputs{
        {4, 5}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {10, 11, 12}};
    std::vector<std::vector<int32_t>> expected_result{
        {23}, {103, 136, 301, 334}, {68, 167, 266, 365}};

    for (size_t i = 0; i < arg0_inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::i32>(shapes[i][0], arg0_inputs[i]),
                           make_host_tensor<element::Type_t::i32>(shapes[i][1], arg1_inputs[i])}));
        EXPECT_EQ(result->get_shape(), (shapes[i][2]));
        ASSERT_EQ(read_vector<int32_t>(result), expected_result[i]);
    }
}

TEST(op_eval, matmul_dynamic_0_elem_arg)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, false, false);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // inner vector contains shapes for arg0, arg1, matmul expected result
    std::vector<std::vector<Shape>> shapes{{Shape{2, 0}, Shape{0, 2}, Shape{2, 2}},
                                           {Shape{0, 2}, Shape{2, 0}, Shape{0, 0}}};

    std::vector<std::vector<float>> arg_inputs{{}, {}};
    std::vector<std::vector<float>> expected_result{{0, 0, 0, 0}, {}};

    for (size_t i = 0; i < arg_inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::f32>(shapes[i][0], arg_inputs[i]),
                           make_host_tensor<element::Type_t::f32>(shapes[i][1], arg_inputs[i])}));
        EXPECT_EQ(result->get_shape(), (shapes[i][2]));
        ASSERT_EQ(read_vector<float>(result), expected_result[i]);
    }
}

TEST(op_eval, matmul_dynamic_2D_args)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, false, false);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // tensor shape for arg0, arg1, matmul result
    std::vector<Shape> shapes = {Shape{3, 2}, Shape{2, 4}, Shape{3, 4}};
    std::vector<float> arg0_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> arg1_data{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    std::vector<float> expected_result{
        8.f, 11.f, 14.f, 17.f, 16.f, 23.f, 30.f, 37.f, 24.f, 35.f, 46.f, 57.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(shapes[0], arg0_data),
                               make_host_tensor<element::Type_t::f32>(shapes[1], arg1_data)}));
    EXPECT_EQ(result->get_shape(), shapes[2]);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
}

TEST(op_eval, matmul_dynamic_2D_transpose0)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, true, false);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // tensor shapes for arg0, arg1, matmul result
    std::vector<Shape> shapes = {Shape{3, 2}, Shape{3, 1}, Shape{2, 1}};
    std::vector<float> arg0_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> arg1_data{0.f, 1.f, 2.f};
    std::vector<float> expected_result{13.f, 16.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(shapes[0], arg0_data),
                               make_host_tensor<element::Type_t::f32>(shapes[1], arg1_data)}));
    EXPECT_EQ(result->get_shape(), shapes[2]);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
}

TEST(op_eval, matmul_dynamic_2D_transpose1)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, false, true);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // tensor shapes for arg0, arg1, matmul result
    std::vector<Shape> shapes = {Shape{3, 2}, Shape{3, 2}, Shape{3, 3}};
    std::vector<float> arg0_data{1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    std::vector<float> arg1_data{2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    std::vector<float> expected_result{8.f, 14.f, 20.f, 18.f, 32.f, 46.f, 28.f, 50.f, 72.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(shapes[0], arg0_data),
                               make_host_tensor<element::Type_t::f32>(shapes[1], arg1_data)}));
    EXPECT_EQ(result->get_shape(), shapes[2]);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
}

TEST(op_eval, matmul_dynamic_same_batch_size)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::f32, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, false, false);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // tensor shapes for arg0, arg1, matmul result
    std::vector<Shape> shapes = {Shape{3, 2, 2, 2}, Shape{3, 2, 2, 1}, Shape{3, 2, 2, 1}};
    std::vector<float> arg0_data(shape_size(shapes[0]));
    std::vector<float> arg1_data(shape_size(shapes[1]));

    // arg0_data is 1.f, 2.f, 3.f, ..., 24.f
    iota(arg0_data.begin(), arg0_data.end(), 1.f);
    // arg1_data is 0.f, 1.f, 2.f, ..., 11.f
    iota(arg1_data.begin(), arg1_data.end(), 0.f);

    std::vector<float> expected_result{
        2.f, 4.f, 28.f, 38.f, 86.f, 104.f, 176.f, 202.f, 298.f, 332.f, 452.f, 494.f};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::f32>(shapes[0], arg0_data),
                               make_host_tensor<element::Type_t::f32>(shapes[1], arg1_data)}));
    EXPECT_EQ(result->get_shape(), shapes[2]);
    ASSERT_TRUE(test::all_close_f(read_vector<float>(result), expected_result));
}

TEST(op_eval, matmul_dynamic_broadcast)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, false, false);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // inner vector contains shapes for arg0, arg1, matmul expected result
    std::vector<std::vector<Shape>> shapes{
        {Shape{2, 1}, Shape{2, 2, 1, 2}, Shape{2, 2, 2, 2}},
        {Shape{2, 1, 1, 2}, Shape{2, 1}, Shape{2, 1, 1, 1}},
        {Shape{2, 2, 1}, Shape{1, 1, 2}, Shape{2, 2, 2}},
        {Shape{1, 1, 2}, Shape{3, 2, 1}, Shape{3, 1, 1}},
        {Shape{1, 2, 1, 2}, Shape{2, 1, 2, 1}, Shape{2, 2, 1, 1}},
        {Shape{1, 2, 3}, Shape{1, 1, 3, 2}, Shape{1, 1, 2, 2}},
        {Shape{4, 1, 2}, Shape{1, 1, 2, 1}, Shape{1, 4, 1, 1}}};

    std::vector<std::vector<int64_t>> arg0_inputs{{2, 3},
                                                  {1, 2, 3, 4},
                                                  {2, 3, 4, 5},
                                                  {1, 2},
                                                  {0, 1, 2, 3},
                                                  {0, 1, 2, 3, 4, 5},
                                                  {0, 1, 2, 3, 4, 5, 6, 7}};
    std::vector<std::vector<int64_t>> arg1_inputs{{0, 1, 2, 3, 4, 5, 6, 7},
                                                  {2, 3},
                                                  {4, 5},
                                                  {1, 2, 3, 4, 5, 6},
                                                  {2, 3, 4, 5},
                                                  {2, 3, 4, 5, 6, 7},
                                                  {0, 1}};
    std::vector<std::vector<int64_t>> expected_result{
        {0, 2, 0, 3, 4, 6, 6, 9, 8, 10, 12, 15, 12, 14, 18, 21},
        {8, 18},
        {8, 10, 12, 15, 16, 20, 20, 25},
        {5, 11, 17},
        {3, 13, 5, 23},
        {16, 19, 52, 64},
        {1, 3, 5, 7}};

    for (size_t i = 0; i < arg0_inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::i64>(shapes[i][0], arg0_inputs[i]),
                           make_host_tensor<element::Type_t::i64>(shapes[i][1], arg1_inputs[i])}));
        EXPECT_EQ(result->get_shape(), (shapes[i][2]));
        ASSERT_EQ(read_vector<int64_t>(result), expected_result[i]);
    }
}

TEST(op_eval, matmul_dynamic_broadcast_transpose0)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, true, false);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // inner vector contains shapes for arg0, arg1, matmul expected result
    std::vector<std::vector<Shape>> shapes{{Shape{3, 2}, Shape{2, 1, 3, 2}, Shape{2, 1, 2, 2}},
                                           {Shape{2, 1, 2, 3}, Shape{2, 1}, Shape{2, 1, 3, 1}},
                                           {Shape{2, 3, 2}, Shape{1, 3, 1}, Shape{2, 2, 1}},
                                           {Shape{1, 2, 3}, Shape{3, 2, 1}, Shape{3, 3, 1}}};

    std::vector<std::vector<int64_t>> arg0_inputs{{2, 3, 4, 5, 6, 7},
                                                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                                  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                                  {2, 3, 4, 5, 6, 7}};
    std::vector<std::vector<int64_t>> arg1_inputs{
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, {2, 3}, {4, 5, 6}, {1, 2, 3, 4, 5, 6}};
    std::vector<std::vector<int64_t>> expected_result{{32, 44, 38, 53, 104, 116, 128, 143},
                                                      {9, 14, 19, 39, 44, 49},
                                                      {49, 64, 139, 154},
                                                      {12, 15, 18, 26, 33, 40, 40, 51, 62}};

    for (size_t i = 0; i < arg0_inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::i64>(shapes[i][0], arg0_inputs[i]),
                           make_host_tensor<element::Type_t::i64>(shapes[i][1], arg1_inputs[i])}));
        EXPECT_EQ(result->get_shape(), (shapes[i][2]));
        ASSERT_EQ(read_vector<int64_t>(result), expected_result[i]);
    }
}

TEST(op_eval, matmul_dynamic_broadcast_transpose1)
{
    auto arg0 = make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic());
    auto arg1 = make_shared<op::Parameter>(element::Type_t::i64, PartialShape::dynamic());
    auto matmul = make_shared<op::MatMul>(arg0, arg1, false, true);
    auto fun = make_shared<Function>(OutputVector{matmul}, ParameterVector{arg0, arg1});

    // inner vector contains shapes for arg0, arg1, matmul expected result
    std::vector<std::vector<Shape>> shapes{{Shape{3, 2}, Shape{2, 1, 3, 2}, Shape{2, 1, 3, 3}},
                                           {Shape{2, 1, 2, 3}, Shape{2, 3}, Shape{2, 1, 2, 2}},
                                           {Shape{2, 3, 2}, Shape{1, 1, 2}, Shape{2, 3, 1}},
                                           {Shape{1, 2, 3}, Shape{3, 1, 3}, Shape{3, 2, 1}}};

    std::vector<std::vector<int64_t>> arg0_inputs{{2, 3, 4, 5, 6, 7},
                                                  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                                  {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
                                                  {2, 3, 4, 5, 6, 7}};
    std::vector<std::vector<int64_t>> arg1_inputs{{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11},
                                                  {2, 3, 4, 5, 6, 7},
                                                  {5, 6},
                                                  {1, 2, 3, 4, 5, 6, 7, 8, 9}};
    std::vector<std::vector<int64_t>> expected_result{
        {3, 13, 23, 5, 23, 41, 7, 33, 59, 33, 43, 53, 59, 77, 95, 85, 111, 137},
        {11, 20, 38, 74, 65, 128, 92, 182},
        {17, 39, 61, 83, 105, 127},
        {20, 38, 47, 92, 74, 146}};

    for (size_t i = 0; i < arg0_inputs.size(); i++)
    {
        auto result = make_shared<HostTensor>();
        ASSERT_TRUE(
            fun->evaluate({result},
                          {make_host_tensor<element::Type_t::i64>(shapes[i][0], arg0_inputs[i]),
                           make_host_tensor<element::Type_t::i64>(shapes[i][1], arg1_inputs[i])}));
        EXPECT_EQ(result->get_shape(), (shapes[i][2]));
        ASSERT_EQ(read_vector<int64_t>(result), expected_result[i]);
    }
}
