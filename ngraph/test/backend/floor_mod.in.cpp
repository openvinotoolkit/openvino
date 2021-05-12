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

NGRAPH_TEST(${BACKEND_NAME}, floor_mod_int32)
{
    Shape shape{4};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<int32_t> a{7, -7, 7, -7};
    std::vector<int32_t> b{3, 3, -3, -3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<int32_t>({a, b});
    test_case.add_expected_output<int32_t>(shape, {1, 2, -2, -1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, floor_mod_int64)
{
    Shape shape{4};

    auto A = make_shared<op::Parameter>(element::i32, shape);
    auto B = make_shared<op::Parameter>(element::i32, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<int32_t> a{7, -7, 7, -7};
    std::vector<int32_t> b{3, 3, -3, -3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<int32_t>({a, b});
    test_case.add_expected_output<int32_t>(shape, {1, 2, -2, -1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, floor_mod_float)
{
    Shape shape{4};

    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<float> a{7, -7, 7, -7};
    std::vector<float> b{3, 3, -3, -3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {1, 2, -2, -1});
    test_case.run();
}

NGRAPH_TEST(${BACKEND_NAME}, floor_mod_broadcasted)
{
    Shape shape_a{2, 1, 2};
    Shape shape_b{2, 1};
    Shape shape_r{2, 2, 2};

    auto A = make_shared<op::Parameter>(element::f32, shape_a);
    auto B = make_shared<op::Parameter>(element::f32, shape_b);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<float> a{1, 2, 3, 4};
    std::vector<float> b{2, 3};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape_r, {1.0f, 0.0f, 1.0f, 2.0f,
                                                   1.0f, 0.0f, 0.0f, 1.0f});
    test_case.run();
}
NGRAPH_TEST(${BACKEND_NAME}, floor_mod_scalars)
{
    Shape shape{};
    auto A = make_shared<op::Parameter>(element::f32, shape);
    auto B = make_shared<op::Parameter>(element::f32, shape);
    auto f = make_shared<Function>(make_shared<op::FloorMod>(A, B), ParameterVector{A, B});

    std::vector<float> a{2};
    std::vector<float> b{4};

    auto test_case = test::TestCase<TestEngine>(f);
    test_case.add_multiple_inputs<float>({a, b});
    test_case.add_expected_output<float>(shape, {2.0f});
    test_case.run();
}
