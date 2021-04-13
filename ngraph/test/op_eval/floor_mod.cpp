// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "ngraph/op/floor_mod.hpp"
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/interpreter_engine.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

TEST(op_eval, floor_mod)
{
    Shape shape{4};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<float> a{5.1, -5.1, 5.1, -5.1};
    std::vector<float> b{3.0, 3.0, -3.0, -3.0};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {2.1, 0.9, -0.9, -2.1});
    test_case.run();
}

<<<<<<< HEAD

TEST(op_eval, floor_mod_broadcasted)
{
    Shape shape_a{2, 1, 2};
    Shape shape_b{2, 1};
    Shape shape_r{2, 2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{2, 3};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_r, {1.0f, 0.0f, 1.0f, 2.0f,
                                                   1.0f, 0.0f, 0.0f, 1.0f});
    test_case.run();
}

TEST(op_eval, floor_mod_scalars)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<float> a{2};
    std::vector<float> b{3};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {2.0f});
    test_case.run();
}

TEST(op_eval, floor_mod_vector_scalar)
{
    Shape shape_a{2, 2};
    Shape shape_b{};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<float> a{2, 3, 4, 5};
    std::vector<float> b{2};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {0.0f, 1.0f, 0.0f, 1.0f});
    test_case.run();
}

TEST(op_eval, floor_mod_int64)
{
    Shape shape{4};

    auto A = make_shared<op::Parameter>(element::i64, shape);
    auto B = make_shared<op::Parameter>(element::i64, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<int64_t> a{5, -5, 5, -5};
    std::vector<int64_t> b{3, 3, -3, -3};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_multiple_inputs<int64_t>({a, b});
    test_case.add_expected_output<int64_t>(shape, {2, -2, 2, -2});
    test_case.run();
}

TEST(op_eval, floor_mod_broadcasted_int64)
{
    Shape shape_a{2, 1, 2};
    Shape shape_b{2, 1};
    Shape shape_r{2, 2, 2};

    auto A = make_shared<op::Parameter>(element::i64, shape_a);
    auto B = make_shared<op::Parameter>(element::i64, shape_b);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<int64_t> a{1, 2, 3, 4};
    std::vector<int64_t> b{2, 3};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_multiple_inputs<int64_t>({a, b});
    test_case.add_expected_output<int64_t>(shape_r, {1, 0, 1, 2,
                                                     1, 0, 0, 1});
    test_case.run();
}

TEST(op_eval, floor_mod_int32)
{
    Shape shape{4};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<int32_t> a{5, -5, 5, -5};
    std::vector<int32_t> b{3, 3, -3, -3};

    auto test_case = test::TestCase<ngraph::test::INTERPRETER_Engine>(f);
    test_case.add_multiple_inputs<int32_t>({a, b});
    test_case.add_expected_output<int32_t>(shape, {2, -2, 2, -2});
    test_case.run();
}

TEST(op_eval, floor_mod_i32)
{
    auto a = make_shared<op::Parameter>(element::i32, Shape{6});
    auto b = make_shared<op::Parameter>(element::i32, Shape{6});
    auto floor_mod = make_shared<op::v1::FloorMod>(a, b);
    auto fun = make_shared<Function>(OutputVector{floor_mod}, ParameterVector{a, b});

    std::vector<int32_t> a_value{-4, 7, 5, 4, -7, 8};
    std::vector<int32_t> b_value{2, -3, 8, -2, 3, 5};
    std::vector<int32_t> expected_result{0, -2,  5,  0,  2,  3};

    auto result = make_shared<HostTensor>();
    ASSERT_TRUE(fun->evaluate({result},
                              {make_host_tensor<element::Type_t::i32>(Shape{6}, a_value),
                               make_host_tensor<element::Type_t::i32>(Shape{6}, b_value)}));
    EXPECT_EQ(result->get_element_type(), element::i32);
    EXPECT_EQ(result->get_shape(), Shape{6});
    auto result_data = read_vector<int32_t>(result);
    for (size_t i = 0; i < expected_result.size(); i++)
        EXPECT_NEAR(result_data[i], expected_result[i], 0.000001);
}

