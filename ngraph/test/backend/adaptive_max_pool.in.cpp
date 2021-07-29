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

NGRAPH_TEST(${BACKEND_NAME}, adaptive_max_pool_1d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 3, 7});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{1}, {3});
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, output_shape);
    auto fun = make_shared<Function>(adaptive_pool->outputs(), ParameterVector{data});

    std::vector<float> inputs{ 0,  4,  1,  3, -2, -5, -2,
                              -2,  1, -3,  1, -3, -4,  0,
                              -2,  1, -1, -2,  3, -1, -3,

                              -1, -2,  3,  4, -3, -4,  1,
                               2,  0, -4, -5, -2, -2, -3,
                               2,  3,  1, -5,  2, -4, -2};
    std::vector<float> expected_result{ 4,  3, -2,
                                        1,  1,  0,
                                        1,  3,  3,

                                        3,  4,  1,
                                        2, -2, -2,
                                        3,  2,  2};
    std::vector<int64_t> expected_indices{1, 3, 4,
                                          1, 3, 6,
                                          1, 4, 4,

                                          2, 3, 6,
                                          0, 4, 4,
                                          1, 4, 4};
    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(Shape{2, 3, 7}, inputs);
    test_case.add_expected_output<float>(Shape{2, 3, 3}, expected_result);
    test_case.add_expected_output<int64_t>(Shape{2, 3, 3}, expected_indices);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, adaptive_max_pool_2d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{1, 3, 7, 10});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{2}, {3, 3});
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, output_shape);
    auto fun = make_shared<Function>(adaptive_pool->outputs(), ParameterVector{data});

    std::vector<float> inputs{ 0, -2, -5, -5,  2,  3,  2, -3,  1, -2,
                              -4, -1, -1, -1,  2, -4,  3, -5, -1, -1,
                               1,  2,  4, -2, -3, -2,  0, -5,  2, -4,
                              -1, -4,  4,  2,  1, -2,  2, -3,  0,  1,
                              -3,  3, -1,  4,  0,  2,  0,  3,  4, -4,
                               1,  4, -1, -5, -2,  4, -3,  3,  2,  1,
                               0,  4,  2, -5,  2, -5, -2, -1,  4,  2,

                               0,  4, -2,  0, -5, -3,  4, -4, -2, -2,
                               2,  1,  4,  3,  2, -5, -4, -4,  0,  1,
                               4, -4, -3,  3,  3,  4, -2, -3, -4, -2,
                               0,  1, -1,  3, -2,  2,  0, -3, -1, -1,
                               0,  0,  2,  2, -2,  1, -3,  1,  2,  4,
                               3, -5, -4,  1, -4,  2,  0, -2, -5,  2,
                              -3, -2, -3, -4,  2, -2, -4,  2, -4, -3,

                               1, -5, -1, -5,  2,  1,  3,  4,  3,  0,
                              -5,  4, -3, -4, -1,  2, -4,  2,  0, -5,
                              -3,  0,  2, -3, -5,  3, -2, -1, -5, -4,
                              -5,  0, -5, -1, -3,  3,  3, -4, -3, -4,
                              -5,  4, -1,  1, -1, -4,  1, -3, -4, -1,
                              -2, -3, -5,  2,  2, -5,  1,  1, -5, -4,
                               0,  2,  4,  2,  0,  2,  4,  0, -5,  2};
    std::vector<float> expected_result{4, 3, 3,
                                       4, 4, 4,
                                       4, 4, 4,

                                       4, 4, 4,
                                       4, 4, 4,
                                       3, 2, 4,

                                       4, 3, 4,
                                       4, 3, 3,
                                       4, 4, 4};
    std::vector<int64_t> expected_indices{22, 5 , 16,
                                          22, 43, 48,
                                          43, 43, 48,

                                          1 , 6 , 6 ,
                                          20, 25, 49,
                                          50, 43, 49,

                                          11, 6 , 7 ,
                                          41, 25, 36,
                                          41, 66, 66};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(Shape{1, 3, 7, 10}, inputs);
    test_case.add_expected_output<float>(Shape{1, 3, 3, 3}, expected_result);
    test_case.add_expected_output<int64_t>(Shape{1, 3, 3, 3}, expected_indices);
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, adaptive_max_pool_3d)
{
    auto data = make_shared<op::Parameter>(element::f32, Shape{2, 2, 3, 3, 3});
    auto output_shape = op::Constant::create<int64_t>(element::i64, Shape{3}, {2, 2, 2});
    auto adaptive_pool = make_shared<op::v8::AdaptiveMaxPool>(data, output_shape);
    auto fun = make_shared<Function>(adaptive_pool->outputs(), ParameterVector{data});

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
    std::vector<float> expected_result{4, 4, 4, 4,
                                       3, 3, 4, 3,

                                       4, 4, 4, 4,
                                       4, 4, 4, 4,

                                       4, 3, 4, 3,
                                       4, 3, 4, 3,

                                       3, 1, 4, 4,
                                       3, 3, 4, 3};
    std::vector<int64_t> expected_indices{4 , 4 , 4 , 4 ,
                                          22, 22, 24, 22,

                                          3 , 14, 3 , 8 ,
                                          18, 14, 22, 14,

                                          0 , 13, 12, 13,
                                          12, 13, 12, 13,

                                          3 , 2 , 7 , 7 ,
                                          22, 22, 24, 22};

    auto test_case = test::TestCase<TestEngine>(fun);
    test_case.add_input<float>(Shape{2, 2, 3, 3, 3}, inputs);
    test_case.add_expected_output<float>(Shape{2, 2, 2, 2, 2}, expected_result);
    test_case.add_expected_output<int64_t>(Shape{2, 2, 2, 2, 2}, expected_indices);
    test_case.run();
}
