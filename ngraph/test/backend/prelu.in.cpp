// Co pyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d)
{
    Shape shape_a{2, 6};
    Shape shape_b{6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, 5, 6,
                         1, 2, -3, -4, 5, 6};
    std::vector<float> b{2, 2, 2, 2, 2, 2};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -6, -8, 5, 6,
                                                   1, 2, -6, -8, 5, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d)
{
    Shape shape_a{6};
    Shape shape_b{1};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, 5, 6};
    std::vector<float> b{2};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -6, -8, 5, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu)
{
    Shape shape{3, 2};
    Shape rshape{2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> b{0, 1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{0, 3, 0, 1, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_shared_slope)
{
    Shape shape{3, 2};
    Shape rshape{2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, 1, -1, -1};
    std::vector<float> b{0.5, 2};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{-1, 3, -1, 1, -0.5, -2});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_slope)
{
    Shape shape{3, 2};
    Shape rshape{2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, rshape);
    auto prelu = make_shared<op::PRelu>(A, B);
    auto f = make_shared<Function>(NodeVector{prelu}, ParameterVector{A, B});
    std::vector<float> a{-2, 3, -2, -1, -1, 0};
    std::vector<float> b{-0.5, -1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(vector<float>{1, 3, 1, 1, 0.5, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_prelu_2d_same_shape)
{
    Shape shape_a{2, 6};
    Shape shape_b{2, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, 5, 6,
                         1, 2, -3, -4, 5, 6};
    std::vector<float> b{2, 2, 2, 2, 2, 2,
                         1, 1, 4, 2, 1, 1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -6, -8, 5, 6,
                                                   1, 2, -12, -8, 5, 6});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_prelu_2d_diff_shape)
{
    Shape shape_a{2, 2, 2, 2};
    Shape shape_b{2, 1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4};
    std::vector<float> b{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -3, -8, 1, 2, -9, -16, 1, 2, -3, -8, 1, 2, -9, -16});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_prelu_3d_diff_shape)
{
    Shape shape_a{2, 2, 6};
    Shape shape_b{2, 1, 6};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, -5, 6,
                         -1, -2, -3, -4, -5, -6,
                         1, 2, -3, -4, 5, 6,
                         -2, 4, -6, -8, 10, 12};
    std::vector<float> b{2, 1, 3, 4, 1, 7,
                         1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -9, -16, -5, 6,
                                                   -2, -2, -9, -16, -5, -42,
                                                   1, 2, -9, -16, 5, 6,
                                                   -2, 4, -18, -32, 10, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, DISABLED_prelu_3d_same_shape)
{
    Shape shape_a{2, 3, 2};
    Shape shape_b{2, 3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, -3, -4, -5, 6,
                         -1, -2, -3, -4,-5, -6,};
    std::vector<float> b{2, 1, 3, 4, 1, 7,
                         1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {1, 2, -9, -16, -5, 6,
                                                   -1, -4, -9, -16, -25, -36});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_broadcast_slope)
{
    Shape shape_a{1, 2, 1, 2};
    Shape shape_b{2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{-10, -10, -10, -10};
    std::vector<float> b{0.1, 10};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {-1, -1, -100, -100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_3d_broadcast_slope)
{
    Shape shape_a{1, 5, 1, 1};
    Shape shape_b{5};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, B), ParameterVector{A, B});

    std::vector<float> a{-1, 0, -1, -1, -1};
    std::vector<float> b{1, 2, 3, 4, 5};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {-1, 0, -3, -4, -5});
    test_case.run();
}
