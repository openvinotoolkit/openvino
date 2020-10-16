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

// Trivial case with no reduced axes.
NGRAPH_TEST(${BACKEND_NAME}, min_trivial)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape, {1, 2, 3, 4});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

// Failure has been reported at 5D for some reason
NGRAPH_TEST(${BACKEND_NAME}, min_trivial_5d)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), ParameterVector{A});
    std::vector<float> a{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                         1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_trivial_5d_int32)
{
    Shape shape{2, 2, 2, 2, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{}), ParameterVector{A});

    std::vector<int32_t> a{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({a});
    test_case.add_expected_output<int32_t>(shape, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_to_scalar)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(Shape{}, {1});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_to_scalar_int8)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::i8, shape);
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    std::vector<int8_t> a{1, 2, 3, 4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int8_t>({a});
    test_case.add_expected_output<int8_t>(Shape{}, {1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_columns)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {1, 2});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    std::vector<float> a{1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {1, 3, 5});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows_int32)
{
    Shape shape_a{3, 2};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    std::vector<int32_t> a{1, 2, 3, 4, 5, 6};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({a});
    test_case.add_expected_output<int32_t>(shape_rt, {1, 3, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_rows_zero)
{
    Shape shape_a{3, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    std::vector<float> a{};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt,
                                         {std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::infinity(),
                                          std::numeric_limits<float>::infinity()});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_cols_zero)
{
    // Now the reduction (g(x:float32[2,2],y:float32[]) = reduce(x,y,f,axes={})).
    Shape shape_a{0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    std::vector<float> a{};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(
        shape_rt, {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_vector_zero)
{
    Shape shape_a{0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    std::vector<float> a{};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {std::numeric_limits<float>::infinity()});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_matrix_to_scalar_zero_by_zero)
{
    Shape shape_a{0, 0};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    std::vector<float> a{};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {std::numeric_limits<float>::infinity()});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_matrix_most_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0}), ParameterVector{A});

    std::vector<float> a{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {1, 2, 3, 4, 5, 6, 7, 8, 9});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_matrix_least_sig)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{2}), ParameterVector{A});

    std::vector<float> a{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {1, 4, 7, 10, 13, 16, 19, 22, 25});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_vector)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1}), ParameterVector{A});

    std::vector<float> a{1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                         15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {1, 2, 3});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_scalar)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    std::vector<float> a{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                         13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {1});
    test_case.run(MIN_FLOAT_TOLERANCE_BITS);
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_to_scalar_int32)
{
    Shape shape_a{3, 3, 3};
    auto A = make_shared<op::Parameter>(element::i32, shape_a);
    Shape shape_rt{};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{0, 1, 2}), ParameterVector{A});

    std::vector<int32_t> a{1,  2,  3,  4,  5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                           13, 12, 11, 10, 9, 8, 7, 6, 5, 4,  3,  2,  1};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<int32_t>({a});
    test_case.add_expected_output<int32_t>(shape_rt, {1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, min_3d_eliminate_zero_dim)
{
    Shape shape_a{3, 0, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    Shape shape_rt{3, 2};
    auto f = make_shared<Function>(make_shared<op::Min>(A, AxisSet{1}), ParameterVector{A});

    std::vector<float> a{};

    float inf = std::numeric_limits<float>::infinity();

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_input<float>({a});
    test_case.add_expected_output<float>(shape_rt, {inf, inf, inf, inf, inf, inf});
    test_case.run();
}
