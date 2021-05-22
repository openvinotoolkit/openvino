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

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

static string s_manifest = "${MANIFEST}";
using TestEngine = test::ENGINE_CLASS_NAME(${BACKEND_NAME});

NGRAPH_TEST(${BACKEND_NAME}, power)
{
    Shape shape{2, 2};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Power>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, 3, 5};
    std::vector<float> b{2, 0, 6, 3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {1, 1, 729, 125});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, power_broadcasted)
{
    Shape shape_a{2, 1, 5};
    Shape shape_b{2, 1};
    Shape shape_r{2, 2, 5};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Power>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<float> b{1, 2};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_r, {1, 2, 3, 4, 5, 1, 4, 9, 16, 25,
                                                   6, 7, 8, 9, 10, 36, 49, 64, 81, 100});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, power_scalars)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::v1::Power>(A, B), ParameterVector{A, B});

    std::vector<float> a{2};
    std::vector<float> b{3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {8});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, power_vector_and_scalar)
{
    Shape shape_a{2, 2};
    Shape shape_b{};
    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::v1::Power>(A, B), ParameterVector{A, B});

    std::vector<float> a{2, 3, 4, 5};
    std::vector<float> b{2};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_a, {4, 9, 16, 25});
    test_case.run();
}