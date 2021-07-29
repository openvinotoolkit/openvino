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

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_1_param)
{
    Shape shape_a{6};
    Shape shape_slope{1};

    std::vector<float> a{1, 2, -3, -4, 5, 6};
    std::vector<float> slope{2};
    std::vector<float> out{1, 2, -6, -8, 5, 6};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_1_const)
{
    Shape shape_a{6};
    Shape shape_slope{1};

    std::vector<float> a{1, 2, -3, -4, 5, 6};
    std::vector<float> slope{2};
    std::vector<float> out{1, 2, -6, -8, 5, 6};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_param)
{
    Shape shape_a{6};
    Shape shape_slope{6};

    std::vector<float> a{1, 2, -3, -4, 5, 6};
    std::vector<float> slope{2, 3, 4, 5, 6, 7};
    std::vector<float> out{1, 2, -12, -20, 5, 6};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_6_const)
{
    Shape shape_a{6};
    Shape shape_slope{6};

    std::vector<float> a{1, 2, -3, -4, 5, 6};
    std::vector<float> slope{2, 3, 4, 5, 6, 7};
    std::vector<float> out{1, 2, -12, -20, 5, 6};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_2_W_param)
{
    Shape shape_a{3, 2};
    Shape shape_slope{2};

    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> slope{0, 1};
    std::vector<float> out{0, 3, 0, 1, 0, 0};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_2_W_const)
{
    Shape shape_a{3, 2};
    Shape shape_slope{2};

    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> slope{0, 1};
    std::vector<float> out{0, 3, 0, 1, 0, 0};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_6_W_param)
{
    Shape shape_a{2, 6};
    Shape shape_slope{6};

    std::vector<float> a{1, 2, -3, -4, 5, 6,
                         7, 8, -9, -10, 11, 12};
    std::vector<float> slope{1, 2, 3, 4, 5, 6};
    std::vector<float> out{1, 2, -9, -16, 5, 6,
                            7, 8, -27, -40, 11, 12};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_6_W_const)
{
    Shape shape_a{2, 6};
    Shape shape_slope{6};

    std::vector<float> a{1, 2, -3, -4, 5, 6,
                         7, 8, -9, -10, 11, 12};
    std::vector<float> slope{1, 2, 3, 4, 5, 6};
    std::vector<float> out{1, 2, -9, -16, 5, 6,
                            7, 8, -27, -40, 11, 12};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_C_2_param)
{
    Shape shape_a{3, 2};
    Shape shape_slope{2};

    std::vector<float> a{-1, -1, -1, -1, -1, -1};
    std::vector<float> slope{2, 0.5};
    std::vector<float> out{-2, -0.5, -2, -0.5, -2, -0.5};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_C_2_const)
{
    Shape shape_a{3, 2};
    Shape shape_slope{2};

    std::vector<float> a{-1, -1, -1, -1, -1, -1};
    std::vector<float> slope{2, 0.5};
    std::vector<float> out{-2, -0.5, -2, -0.5, -2, -0.5};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_equal_dims_slope_param)
{
    Shape shape_a{2, 2, 2};
    Shape shape_slope{2};

    std::vector<float> a{-0.5, -2, -3, -4, -5, -6, -7, -8};
    std::vector<float> slope{-0.5, -2};
    // std::vector<float> out{0.25, 4, 1.5, 8, 2.5, 12, 3.5, 16}; // broadcast (1, 1, 2)
    std::vector<float> out{0.25, 1, 6, 8, 2.5, 3, 14, 16}; // broadcast (1, 2, 1)
    // std::vector<float> out{0.25, 1, 1.5, 2, 10, 12, 14, 16}; // broadcast (2, 1, 1)

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_equal_dims_slope_const)
{
    Shape shape_a{2, 2, 2};
    Shape shape_slope{2};

    std::vector<float> a{-0.5, -2, -3, -4, -5, -6, -7, -8};
    std::vector<float> slope{-0.5, -2};
    // std::vector<float> out{0.25, 4, 1.5, 8, 2.5, 12, 3.5, 16}; // broadcast (1, 1, 2)
    std::vector<float> out{0.25, 1, 6, 8, 2.5, 3, 14, 16}; // broadcast (1, 2, 1)
    // std::vector<float> out{0.25, 1, 1.5, 2, 10, 12, 14, 16}; // broadcast (2, 1, 1)

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_slope_param)
{
    Shape shape_a{3, 2};
    Shape shape_slope{2};

    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> slope{-0.5, -1};
    std::vector<float> out{1, 3, 1, 1, 0.5, 0};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_negative_slope_const)
{
    Shape shape_a{3, 2};
    Shape shape_slope{2};

    std::vector<float> a{-2, 3, -2, 1, -1, 0};
    std::vector<float> slope{-0.5, -1};
    std::vector<float> out{1, 3, 1, 1, 0.5, 0};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_same_shape_param)
{
    Shape shape_a{2, 6};
    Shape shape_slope{2, 6};

    std::vector<float> a{1, 2, -3, -4, 5, 6,
                         1, 2, -3, -4, 5, 6};
    std::vector<float> slope{2, 2, 2, 2, 2, 2,
                         1, 1, 4, 2, 1, 1};
    std::vector<float> out{1, 2, -6, -8, 5, 6, 1, 2, -12, -8, 5, 6};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_2d_same_shape_const)
{
    Shape shape_a{2, 6};
    Shape shape_slope{2, 6};

    std::vector<float> a{1, 2, -3, -4, 5, 6,
                         1, 2, -3, -4, 5, 6};
    std::vector<float> slope{2, 2, 2, 2, 2, 2,
                         1, 1, 4, 2, 1, 1};
    std::vector<float> out{1, 2, -6, -8, 5, 6, 1, 2, -12, -8, 5, 6};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_C_1_W_slope_param)
{
    Shape shape_a{2, 2, 2, 2};
    Shape shape_slope{2, 1, 2};

    std::vector<float> a{1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4};
    std::vector<float> slope{1, 2, 3, 4};
    std::vector<float> out{1, 2, -3, -8, 1, 2, -9, -16, 1, 2, -3, -8, 1, 2, -9, -16};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_C_1_W_slope_const)
{
    Shape shape_a{2, 2, 2, 2};
    Shape shape_slope{2, 1, 2};

    std::vector<float> a{1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4};
    std::vector<float> slope{1, 2, 3, 4};
    std::vector<float> out{1, 2, -3, -8, 1, 2, -9, -16, 1, 2, -3, -8, 1, 2, -9, -16};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_4d_slope_1_C_1_W_param)
{
    Shape shape_a{2, 2, 2, 2};
    Shape shape_slope{1, 2, 1, 2};

    std::vector<float> a{1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4};
    std::vector<float> slope{1, 2, 3, 4};
    std::vector<float> out{1, 2, -3, -8, 1, 2, -9, -16, 1, 2, -3, -8, 1, 2, -9, -16};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_4d_slope_1_C_1_W_const)
{
    Shape shape_a{2, 2, 2, 2};
    Shape shape_slope{1, 2, 1, 2};

    std::vector<float> a{1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4, 1, 2, -3, -4};
    std::vector<float> slope{1, 2, 3, 4};
    std::vector<float> out{1, 2, -3, -8, 1, 2, -9, -16, 1, 2, -3, -8, 1, 2, -9, -16};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_3d_W_param)
{
    Shape shape_a{2, 2, 6};
    Shape shape_slope{2, 1, 6};


    std::vector<float> a{1, 2, -3, -4, -5, 6,
                         -1, -2, -3, -4, -5, -6,
                         1, 2, -3, -4, 5, 6,
                         -2, 4, -6, -8, 10, 12};
    std::vector<float> slope{2, 1, 3, 4, 1, 7,
                         1, 2, 3, 4, 5, 6};
    std::vector<float> out{1, 2, -9, -16, -5, 6,
                            -2, -2, -9, -16, -5, -42,
                            1, 2, -9, -16, 5, 6,
                            -2, 4, -18, -32, 10, 12};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_3d_W_const)
{
    Shape shape_a{2, 2, 6};
    Shape shape_slope{2, 1, 6};

    std::vector<float> a{1, 2, -3, -4, -5, 6,
                         -1, -2, -3, -4, -5, -6,
                         1, 2, -3, -4, 5, 6,
                         -2, 4, -6, -8, 10, 12};
    std::vector<float> slope{2, 1, 3, 4, 1, 7,
                         1, 2, 3, 4, 5, 6};
    std::vector<float> out{1, 2, -9, -16, -5, 6,
                            -2, -2, -9, -16, -5, -42,
                            1, 2, -9, -16, 5, 6,
                            -2, 4, -18, -32, 10, 12};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_3d_same_shape_param)
{
    Shape shape_a{2, 3, 2};
    Shape shape_slope{2, 3, 2};

    std::vector<float> a{1, 2, -3, -4, -5, 6,
                         -1, -2, -3, -4,-5, -6,};
    std::vector<float> slope{2, 1, 3, 4, 1, 7,
                            1, 2, 3, 4, 5, 6};
    std::vector<float> out{1, 2, -9, -16, -5, 6,
                            -1, -4, -9, -16, -25, -36};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_3d_same_shape_const)
{
    Shape shape_a{2, 3, 2};
    Shape shape_slope{2, 3, 2};

    std::vector<float> a{1, 2, -3, -4, -5, 6,
                         -1, -2, -3, -4,-5, -6,};
    std::vector<float> slope{2, 1, 3, 4, 1, 7,
                            1, 2, 3, 4, 5, 6};
    std::vector<float> out{1, 2, -9, -16, -5, 6,
                            -1, -4, -9, -16, -25, -36};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_3d_broadcast_C_W_slope_param)
{
    Shape shape_a{2, 1, 2};
    Shape shape_slope{2};

    std::vector<float> a{-10, -10, -10, -10};
    std::vector<float> slope{0.1, 10};
    std::vector<float> out{-1, -100, -1, -100}; // broadcast (1, 2, 1)
    // std::vector<float> out{-1, -1, -100, -100}; // broadcast (2, 1, 1)

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}


NGRAPH_TEST(${BACKEND_NAME}, prelu_3d_broadcast_C_W_slope_const)
{
    Shape shape_a{2, 1, 2};
    Shape shape_slope{2};

    std::vector<float> a{-10, -10, -10, -10};
    std::vector<float> slope{0.1, 10};
    std::vector<float> out{-1, -100, -1, -100}; // broadcast (1, 2, 1)
    // std::vector<float> out{-1, -1, -100, -100}; // broadcast (2, 1, 1)

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_4d_broadcast_C_W_slope_param)
{
    Shape shape_a{1, 2, 1, 2};
    Shape shape_slope{2};

    std::vector<float> a{-10, -10, -10, -10};
    std::vector<float> slope{0.1, 10};
    // std::vector<float> out{-1, -100, -1, -100}; // broadcast (1, 1, 1, 2)
    std::vector<float> out{-1, -1, -100, -100}; // broadcast (1, 2, 1, 1)

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_4d_broadcast_C_W_slope_const)
{
    Shape shape_a{1, 2, 1, 2};
    Shape shape_slope{2};

    std::vector<float> a{-10, -10, -10, -10};
    std::vector<float> slope{0.1, 10};
    // std::vector<float> out{-1, -100, -1, -100}; // broadcast (1, 1, 1, 2)
    std::vector<float> out{-1, -1, -100, -100}; // broadcast (1, 2, 1, 1)

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_4d_broadcast_C_slope_param)
{
    Shape shape_a{1, 5, 1, 1};
    Shape shape_slope{5};

    std::vector<float> a{-1, 0, -1, -1, -1};
    std::vector<float> slope{1, 2, 3, 4, 5};
    std::vector<float> out{-1, 0, -3, -4, -5};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_4d_broadcast_C_slope_const)
{
    Shape shape_a{1, 5, 1, 1};
    Shape shape_slope{5};

    std::vector<float> a{-1, 0, -1, -1, -1};
    std::vector<float> slope{1, 2, 3, 4, 5};
    std::vector<float> out{-1, 0, -3, -4, -5};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_batch_nd_elementwise_param)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{2, 3, 4, 5};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};
    std::vector<float> slope(shape_size(shape_slope));
    std::iota(std::begin(slope), std::end(slope), 0);

    std::vector<float> out{-0.,   -1.,   -2.,   -3.,   -4.,   -5.,   -6.,   -7.,   -8.,
         -9.,  -10.,  -11.,  -12.,  -13.,  -14.,  -15.,  -16.,  -17.,
        -18.,  -19.,  -20.,  -21.,  -22.,  -23.,  -24.,  -25.,  -26.,
        -27.,  -28.,  -29.,  -30.,  -31.,  -32.,  -33.,  -34.,  -35.,
        -36.,  -37.,  -38.,  -39.,  -40.,  -41.,  -42.,  -43.,  -44.,
        -45.,  -46.,  -47.,  -48.,  -49.,  -50.,  -51.,  -52.,  -53.,
        -54.,  -55.,  -56.,  -57.,  -58.,  -59.,  -60.,  -61.,  -62.,
        -63.,  -64.,  -65.,  -66.,  -67.,  -68.,  -69.,  -70.,  -71.,
        -72.,  -73.,  -74.,  -75.,  -76.,  -77.,  -78.,  -79.,  -80.,
        -81.,  -82.,  -83.,  -84.,  -85.,  -86.,  -87.,  -88.,  -89.,
        -90.,  -91.,  -92.,  -93.,  -94.,  -95.,  -96.,  -97.,  -98.,
        -99., -100., -101., -102., -103., -104., -105., -106., -107.,
       -108., -109., -110., -111., -112., -113., -114., -115., -116.,
       -117., -118., -119.};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_batch_nd_elementwise_const)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{2, 3, 4, 5};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};
    std::vector<float> slope(shape_size(shape_slope));
    std::iota(std::begin(slope), std::end(slope), 0);

    std::vector<float> out{-0.,   -1.,   -2.,   -3.,   -4.,   -5.,   -6.,   -7.,   -8.,
         -9.,  -10.,  -11.,  -12.,  -13.,  -14.,  -15.,  -16.,  -17.,
        -18.,  -19.,  -20.,  -21.,  -22.,  -23.,  -24.,  -25.,  -26.,
        -27.,  -28.,  -29.,  -30.,  -31.,  -32.,  -33.,  -34.,  -35.,
        -36.,  -37.,  -38.,  -39.,  -40.,  -41.,  -42.,  -43.,  -44.,
        -45.,  -46.,  -47.,  -48.,  -49.,  -50.,  -51.,  -52.,  -53.,
        -54.,  -55.,  -56.,  -57.,  -58.,  -59.,  -60.,  -61.,  -62.,
        -63.,  -64.,  -65.,  -66.,  -67.,  -68.,  -69.,  -70.,  -71.,
        -72.,  -73.,  -74.,  -75.,  -76.,  -77.,  -78.,  -79.,  -80.,
        -81.,  -82.,  -83.,  -84.,  -85.,  -86.,  -87.,  -88.,  -89.,
        -90.,  -91.,  -92.,  -93.,  -94.,  -95.,  -96.,  -97.,  -98.,
        -99., -100., -101., -102., -103., -104., -105., -106., -107.,
       -108., -109., -110., -111., -112., -113., -114., -115., -116.,
       -117., -118., -119.};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_W_slope_param)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{5};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};

    std::vector<float> slope{0, 1, 2, 3, 4};

    std::vector<float> out{-0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2.,
       -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0.,
       -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3.,
       -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1.,
       -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
       -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2.,
       -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0.,
       -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3.,
       -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1.,
       -2., -3., -4.};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_W_slope_const)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{5};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};

    std::vector<float> slope{0, 1, 2, 3, 4};

    std::vector<float> out{-0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2.,
       -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0.,
       -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3.,
       -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1.,
       -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4.,
       -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2.,
       -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0.,
       -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1., -2., -3.,
       -4., -0., -1., -2., -3., -4., -0., -1., -2., -3., -4., -0., -1.,
       -2., -3., -4.};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_C_slope_param)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{3};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};

    std::vector<float> slope{0, 1, 2};

    std::vector<float> out{-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2.};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_1d_C_slope_const)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{3};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};

    std::vector<float> slope{0, 1, 2};

    std::vector<float> out{-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2.};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    const auto f = make_shared<Function>(make_shared<op::v0::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_C_1_1_slope_param)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{3, 1, 1};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};

    std::vector<float> slope{0, 1, 2};

    std::vector<float> out{-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2.};

    const auto A = make_shared<op::Parameter>(element::f32, shape_a);
    const auto SLOPE = make_shared<op::Parameter>(element::f32, shape_slope);
    const auto f = make_shared<Function>(make_shared<op::PRelu>(A, SLOPE), ParameterVector{A, SLOPE});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, slope});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, prelu_C_1_1_slope_const)
{
    Shape shape_a{2, 3, 4, 5};
    Shape shape_slope{3, 1, 1};

    std::vector<float> a{-1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1.};

    std::vector<float> slope{0, 1, 2};

    std::vector<float> out{-0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -0., -0., -0., -0., -0.,
       -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0., -0.,
       -0., -0., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
       -1., -1., -1., -1., -1., -1., -1., -1., -1., -2., -2., -2., -2.,
       -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2., -2.,
       -2., -2., -2.};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto SLOPE = make_shared<op::Constant>(element::f32, shape_slope, slope);
    auto f = make_shared<Function>(make_shared<op::PRelu>(A, SLOPE), ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a});
    test_case.add_expected_output<float>(shape_a, out);
    test_case.run();
}
