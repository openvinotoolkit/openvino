// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <random>
#include <string>

// clang-format off
#ifdef ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#define DEFAULT_FLOAT_TOLERANCE_BITS ${BACKEND_NAME}_FLOAT_TOLERANCE_BITS
#endif

#ifdef ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#define DEFAULT_DOUBLE_TOLERANCE_BITS ${BACKEND_NAME}_DOUBLE_TOLERANCE_BITS
#endif
// clang-format on

#include "gtest/gtest.h"
#include "ngraph/ngraph.hpp"
#include "util/engine/test_engines.hpp"
#include "util/test_case.hpp"
#include "util/test_control.hpp"

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_no_broadcast)
{
    Shape shape{1, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::SquaredDifference>(A, B), ParameterVector{A, B});

    vector<float> a{256, 56};
    vector<float> b{256, 56};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {0, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_negative_numbers)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::SquaredDifference>(A, B), ParameterVector{A, B});

    vector<float> a{256, 56, -21, -14};
    vector<float> b{-112, 56, 6, -8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {135424, 0, 729, 36});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_broadcast)
{
    Shape shape_a{1, 2};
    Shape shape_b{3, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::SquaredDifference>(A, B), ParameterVector{A, B});

    vector<float> a{1, 2};
    vector<float> b{5, 6, 7, 8, 2, 3, 1, 5, 6, 7, 1, 3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_b, {16, 16, 36, 36, 1, 1, 0, 9, 25, 25, 0, 1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_scalars)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::SquaredDifference>(A, B), ParameterVector{A, B});

    vector<float> a{57};
    vector<float> b{13};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {1936});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_vector_and_scalar)
{
    Shape shape_a{2, 2};
    Shape shape_b{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::SquaredDifference>(A, B), ParameterVector{A, B});

    vector<float> a{2, 4, 7, 8};
    vector<float> b{8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {36, 16, 1, 0});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, squared_difference_in_place)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto T = make_shared<op::SquaredDifference>(A, B);
    auto T2 = make_shared<op::SquaredDifference>(T, T);

    auto f = make_shared<Function>(T2, ParameterVector{A, B});

    vector<float> a{1, 2, 3, 4};
    vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {0, 0 ,0 ,0});
    test_case.run();
} 
