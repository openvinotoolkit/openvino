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
    test_case.add_expected_output<int64_t>(shape, {2, 1, -1, -2});
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
    test_case.add_expected_output<int32_t>(shape, {2, 1, -1, -2});
    test_case.run();
}
