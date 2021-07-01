// Copyright (C) 2018-2021 Intel Corporation
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

NGRAPH_TEST(${BACKEND_NAME}, add)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    vector<float> a{1, 2, 3, 4};
    vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {6, 8, 10, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_overload)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    vector<float> a{1, 2, 3, 4};
    vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {6, 8, 10, 12});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_in_place)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto T = make_shared<op::v1::Add>(A, B);
    auto T2 = make_shared<op::v1::Add>(T, T);
    auto T3 = make_shared<op::v1::Add>(T2, T2);
    auto T4 = make_shared<op::v1::Add>(T3, T3);

    auto f = make_shared<Function>(T4, ParameterVector{A, B});

    vector<float> a{1, 2, 3, 4};
    vector<float> b{5, 6, 7, 8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {48, 64, 80, 96});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_broadcast)
{
    Shape shape_a{1, 2};
    Shape shape_b{3, 2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    vector<float> a{1, 2};
    vector<float> b{5, 6, 7, 8, 2, 3, 1, 5, 6, 7, 1, 3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_b, {6, 8, 8, 10, 3, 5, 2, 7, 7, 9, 2, 5});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_scalars)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    vector<float> a{2};
    vector<float> b{8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {10});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, add_vector_and_scalar)
{
    Shape shape_a{2, 2};
    Shape shape_b{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Add>(A, B), ParameterVector{A, B});

    vector<float> a{2, 4, 7, 8};
    vector<float> b{8};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {10, 12, 15, 16});
    test_case.run();
}
