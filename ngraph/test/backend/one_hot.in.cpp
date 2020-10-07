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

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

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

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_2_in_3)
{
    Shape shape_a{};
    Shape shape_r{3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, shape_r, 0);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({2});
    test_case.add_expected_output<int32_t>(shape_r, {0, 0, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_1_in_3)
{
    Shape shape_a{};
    Shape shape_r{3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, shape_r, 0);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({1});
    test_case.add_expected_output<int32_t>(shape_r, {0, 1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_scalar_0_in_3)
{
    Shape shape_a{};
    Shape shape_r{3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, shape_r, 0);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({0});
    test_case.add_expected_output<int32_t>(shape_r, {1, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_0)
{
    Shape shape_a{8};
    Shape shape_r{3, 8};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, shape_r, 0);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({2, 1, 0, 0, 2, 2, 1, 0});
    test_case.add_expected_output<int32_t>(
        shape_r, {0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1)
{
    Shape shape_a{8};
    Shape shape_r{8, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, shape_r, 1);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({2, 1, 0, 0, 2, 2, 1, 0});
    test_case.add_expected_output<int32_t>(
        shape_r, {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_1_barely_oob)
{
    Shape shape_a{8};
    Shape shape_r{8, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, shape_r, 1);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({2, 1, 0, 0, 3, 2, 1, 0});
    test_case.add_expected_output<int32_t>(
        shape_r, {0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_matrix_0)
{
    Shape shape_a{3, 3};
    Shape shape_r{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, shape_r, 0);
    auto f = make_shared<Function>(r, ParameterVector{A});

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({0, 1, 1, 2, 1, 0, 0, 2, 1});
    test_case.add_expected_output<int32_t>(
        shape_r, {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, one_hot_vector_many_categories)
{
    // Imagenet has roughly 20,000 categories
    constexpr uint32_t category_count = 20000;
    Shape shape_a{6};
    Shape shape_r{6, category_count};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    auto r = make_shared<op::OneHot>(A, Shape{6, category_count}, 1);
    auto f = make_shared<Function>(r, ParameterVector{A});

    vector<int32_t> input{0, 11, 101, 1001, 10001, static_cast<int32_t>(category_count - 1)};
    vector<int32_t> output(shape_size(shape_r), 0);
    for (size_t i = 0; i < input.size(); ++i)
    {
        output[i * category_count + input[i]] = 1;
    }

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>(input);
    test_case.add_expected_output<int32_t>(shape_r, output);
    test_case.run();
}
