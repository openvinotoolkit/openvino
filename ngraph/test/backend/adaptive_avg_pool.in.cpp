// Copyright (C) 2018-2021 Intel Corporation
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

NGRAPH_TEST(${BACKEND_NAME}, adaptive_avg_pool_1d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 7});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{1}, {3});
    auto adaptive_pool = make_shared<op::v8::AdaptiveAvgPool>(data, output_shape);
    auto fun = make_shared<Function>(OutputVector{adaptive_pool}, ParameterVector{data});

    std::vector<float> inputs{ 0,  4,  1,  3, -2, -5, -2,
                              -2,  1, -3,  1, -3, -4,  0,
                              -2,  1, -1, -2,  3, -1, -3,

                              -1, -2,  3,  4, -3, -4,  1,
                               2,  0, -4, -5, -2, -2, -3,
                               2,  3,  1, -5,  2, -4, -2};
    std::vector<float> expected_result{ 1.66666663,  0.66666669, -3.        ,
                                       -1.33333337, -1.66666663, -2.33333325,
                                       -0.66666669,  0.        , -0.33333334,

                                        0.        ,  1.33333337, -2.        ,
                                       -0.66666669, -3.66666675, -2.33333325,
                                        2.        , -0.66666669, -1.33333337};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(Shape{2, 3, 7}, inputs);
    test_case.add_expected_output<float>(Shape{2, 3, 3}, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, adaptive_avg_pool_2d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 7, 10});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {3, 3});
    auto adaptive_pool = make_shared<op::v8::AdaptiveAvgPool>(data, output_shape);
    auto fun = make_shared<Function>(OutputVector{adaptive_pool}, ParameterVector{data});

    std::vector<float> inputs{-2, -3, -4,  3, -5,  4,  0, -4, -2, -4,
                              -5,  0, -3,  0, -2,  0,  0, -5, -4, -1,
                               3, -1,  0, -1,  0, -2,  0,  4,  1,  4,
                               0, -1, -4,  2, -2, -5, -1, -1, -2,  1,
                               2, -2, -1,  2,  0, -1,  0, -5,  4,  4,
                               3,  0, -4, -4, -4, -2,  0,  1, -2, -1,
                               4, -2, -4,  1, -1, -3, -4, -1,  1, -4,

                              -2, -4, -5,  0, -4,  3,  4, -5, -4, -2,
                               0,  2, -4, -3,  3, -1,  1, -4, -5,  4,
                               2, -5,  2, -3,  0,  4,  3,  3,  1,  2,
                              -1, -4,  1, -3, -3, -2,  3,  4, -2, -5,
                               1,  4,  4, -2,  2,  1, -5, -2, -5,  1,
                               1, -2, -3, -3, -1, -5,  1, -3, -5, -3,
                              -4, -1,  4, -3,  4, -1,  4,  3,  1,  4,

                              -2, -4, -4,  4, -3,  4,  2, -3, -2,  4,
                              -3,  0,  1, -4,  4,  4,  0,  3, -1,  3,
                               3, -5,  0,  3, -3,  1, -2,  4, -5, -5,
                               1,  0, -1,  0, -3, -2,  0, -3,  3, -2,
                              -2,  0, -3,  4, -1,  2, -2,  2, -3, -1,
                              -4, -2,  0,  2,  0,  2,  0, -3,  4,  3,
                              -5, -3, -5,  1, -5, -3, -5,  4, -3,  3};
    std::vector<float> expected_result{-1.08333337, -0.25000000, -0.91666669,
                                       -0.08333334, -0.66666669,  0.75000000,
                                       -0.41666666, -1.33333337, -0.58333331,

                                       -1.66666663,  0.58333331, -0.16666667,
                                       -0.33333334, -0.41666666, -0.16666667,
                                       -0.33333334, -0.66666669, -0.75000000,

                                       -0.91666669,  0.83333331, -0.16666667,
                                        0.        , -0.25000000, -1.16666663,
                                       -1.41666663, -0.41666666, -0.08333334};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(Shape{1, 3, 7, 10}, inputs);
    test_case.add_expected_output<float>(Shape{1, 3, 3, 3}, expected_result);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, adaptive_avg_pool_3d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3, 3});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 2, 2});
    auto adaptive_pool = make_shared<op::v8::AdaptiveAvgPool>(data, output_shape);
    auto fun = make_shared<Function>(OutputVector{adaptive_pool}, ParameterVector{data});

    std::vector<float> inputs{-5,  1, -3, -4,  4, -4,  3, -3, -1,
                               0,  0, -2, -4,  2,  0, -4, -5, -2,
                              -4, -4,  0, -2,  3, -3,  4, -1, -4,

                              -1, -1, -5,  4, -1, -2, -3,  0,  4,
                              -1, -5, -4,  1,  1,  4, -5, -5, -5,
                               4, -3, -3, -3,  4,  0, -3, -5,  1,

                               4,  2,  1, -5, -5,  1,  0, -4, -1,
                               2, -4, -2,  4,  3,  1, -3, -3, -2,
                              -4, -3, -3,  3, -1,  1,  2,  2, -4,

                              -5, -4,  1,  3, -4, -1,  2,  4, -5,
                               0,  1, -2,  0,  0, -2,  3, -2, -5,
                              -3, -5, -2, -1,  3, -2,  4,  3, -3};
    std::vector<float> expected_result{-0.750, -0.250, -1.375, -1.125,
                                       -1.125, -0.500, -0.875, -1.250,

                                       -0.375, -1.625, -1.   , -0.500,
                                       -0.250, -0.750, -1.875, -0.625,

                                        0.125, -0.375, -1.625, -1.250,
                                        0.   , -1.   ,  0.875, -0.375,

                                       -1.125, -1.375,  0.750, -1.875,
                                       -0.625, -1.125,  1.250, -1.};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(Shape{2, 2, 3, 3, 3}, inputs);
    test_case.add_expected_output<float>(Shape{2, 2, 2, 2, 2}, expected_result);
    test_case.run();
}
