//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
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

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_no_variance)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, false, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-4, -3, -2, -1, 0, 1, 2, 3, 4,
                                          -4, -3, -2, -1, 0, 1, 2, 3, 4,
                                          -4, -3, -2, -1, 0, 1, 2, 3, 4});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_inside_sqrt)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{2}, vector<int64_t>{2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::INSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>({1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_across_chanells)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{1, 3, 3, 3});
    auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{1, 2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9});

    test_case.add_expected_output<float>({-1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934,
                                          -1.5491934, -1.161895, -0.7745967,
                                          -0.38729835, 0.,        0.38729835,
                                           0.7745967,  1.161895,  1.5491934});
    // clang-format on
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, evaluate_mvn_6_across_batch)
{
    auto data = make_shared<op::Parameter>(element::f32, PartialShape{2, 3, 2, 2});
    auto axes = make_shared<op::Constant>(element::i64, Shape{3}, vector<int64_t>{0, 2, 3});

    auto mvn =
        make_shared<op::v6::MVN>(data, axes, true, 1e-9, ngraph::op::MVNEpsMode::OUTSIDE_SQRT);
    auto fun = make_shared<Function>(OutputVector{mvn}, ParameterVector{data});
    auto test_case = test::TestCase<TestEngine>(fun);

    // clang-format off
    test_case.add_input<float>(
        {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});

    test_case.add_expected_output<float>(
        {-1.5275252,  -1.0910894, -0.65465367, -0.21821788, 0.21821788,  0.65465367,
          1.0910894,   1.5275252, -1.5275252,  -1.0910894, -0.65465367, -0.21821788,
          0.21821788,  0.65465367, 1.0910894,   1.5275252, -1.5275252,  -1.0910894,
         -0.65465367, -0.21821788, 0.21821788,  0.65465367, 1.0910894,   1.5275252});
    // clang-format on
    test_case.run();
}
